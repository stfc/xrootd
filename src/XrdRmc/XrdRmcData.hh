#ifndef __XRDRMCDATA_HH__
#define __XRDRMCDATA_HH__
/******************************************************************************/
/*                                                                            */
/*                         X r d R m c D a t a . h h                          */
/*                                                                            */
/* (c) 2019 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/*                                                                            */
/* This file is part of the XRootD software suite.                            */
/*                                                                            */
/* XRootD is free software: you can redistribute it and/or modify it under    */
/* the terms of the GNU Lesser General Public License as published by the     */
/* Free Software Foundation, either version 3 of the License, or (at your     */
/* option) any later version.                                                 */
/*                                                                            */
/* XRootD is distributed in the hope that it will be useful, but WITHOUT      */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or      */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public       */
/* License for more details.                                                  */
/*                                                                            */
/* You should have received a copy of the GNU Lesser General Public License   */
/* along with XRootD in a file called COPYING.LESSER (LGPL license) and file  */
/* COPYING (GPL license).  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                            */
/* The copyright holder's institutional names and contributor's names may not */
/* be used to endorse or promote products derived from this software without  */
/* specific prior written permission of the institution or contributor.       */
/******************************************************************************/

/* The XrdRmcData object defines a remanufactured XrdOucCacheIO object and
   is used to front a XrdOucCacheIO object with an XrdRmcReal object.
*/

#include "XrdOuc/XrdOucCache.hh"
#include "XrdRmc/XrdRmcReal.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysXSLock.hh"

class XrdRmcData : public XrdOucCacheIO
{
public:

bool           Detach(XrdOucCacheIOCD &iocd);

long long      FSize() {return (ioObj ? ioObj->FSize() : 0);}

const char    *Path() {return ioObj->Path();}

void           Preread();

void           Preread(aprParms &Parms);

void           Preread(long long Offs, int rLen, int Opts=0);

int            Read (char  *Buffer, long long  Offset, int  Length);

static int     setAPR(aprParms &Dest, aprParms &Src, int pSize);

int            Sync() {return 0;} // We only support write-through for now

int            Trunc(long long Offset);

int            Write(char  *Buffer, long long  Offset,  int  Length);

               XrdRmcData(XrdRmcReal *cP, XrdOucCacheIO *ioP,
                           long long    vn, int            opts);

private:
              ~XrdRmcData() {}
void           QueuePR(long long SegOffs, int rLen, int prHow, int isAuto=0);
int            Read (XrdOucCacheStats &Now,
                     char *Buffer, long long Offs, int Length);

using XrdOucCacheIO::Read;

// The following is for read/write support
//
class MrSw
{
public:
inline void UnLock() {if (myLock) {myLock->UnLock(myUsage); myLock = 0;}}

            MrSw(XrdSysXSLock *lP, XrdSysXS_Type usage) : myUsage(usage)
                {if ((myLock = lP)) lP->Lock(usage);}
           ~MrSw() {if (myLock) myLock->UnLock(myUsage);}

private:
XrdSysXSLock *myLock;
XrdSysXS_Type myUsage;
};

// Statics per connection
//
XrdOucCacheStats Statistics;

// The following supports MRSW serialization
//
XrdSysXSLock     rwLock;
XrdSysXSLock    *pPLock;  // 0 if no preread lock required
XrdSysXSLock    *rPLock;  // 0 if no    read lock required
XrdSysXSLock    *wPLock;  // 0 if no   write lock required
XrdSysXS_Type    pPLopt;
XrdSysXS_Type    rPLopt;

XrdSysMutex      DMutex;
XrdRmcReal     *Cache;
XrdOucCacheIO   *ioObj;
long long        VNum;
long long        SegSize;
long long        OffMask;
long long        SegShft;
int              maxCache;
char             isFIS;
char             isRW;
char             Debug;

static const int okRW   = 1;
static const int xqRW   = 2;

// Preread Control Area
//
XrdRmcReal::prTask prReq;
XrdSysSemaphore    *prStop;

long long        prNSS;          // Next Sequential Segment for maxi prereads

static const int prRRMax= 5;
long long        prRR[prRRMax];  // Recent reads
int              prRRNow;        // Pointer to next entry to use

static const int prMax  = 8;
static const int prRun  = 1;     // Status in prActive (running)
static const int prWait = 2;     // Status in prActive (waiting)

static const int prLRU  = 1;     // Status in prOpt    (set LRU)
static const int prSUSE = 2;     // Status in prOpt    (set Single Use)
static const int prSKIP = 3;     // Status in prOpt    (skip entry)

aprParms         Apr;
long long        prCalc;
long long        prBeg[prMax];
long long        prEnd[prMax];
int              prNext;
int              prFree;
int              prPerf;
char             prOpt[prMax];
char             prOK;
char             prActive;
char             prAuto;
};
#endif
