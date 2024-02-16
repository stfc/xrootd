#ifndef __CRYPTO_SSLX509CRL_H__
#define __CRYPTO_SSLX509CRL_H__
/******************************************************************************/
/*                                                                            */
/*                X r d C r y p t o s s l X 5 0 9 C r l . h h                 */
/*                                                                            */
/* (c) 2005 G. Ganis , CERN                                                   */
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
/*                                                                            */
/******************************************************************************/
#include <openssl/x509v3.h>

/* ************************************************************************** */
/*                                                                            */
/* OpenSSL X509 CRL implementation        .                                   */
/*                                                                            */
/* ************************************************************************** */

#include "XrdSut/XrdSutCache.hh"
#include "XrdCrypto/XrdCryptoX509Crl.hh"

// ---------------------------------------------------------------------------//
//
// X509 CRL interface
// Describes one CRL certificate
//
// ---------------------------------------------------------------------------//

class XrdCryptoX509;

class XrdCryptosslX509Crl : public XrdCryptoX509Crl {
public:

   XrdCryptosslX509Crl(const char *crlf, int opt = 0);
   XrdCryptosslX509Crl(FILE *, const char *crlf);
   XrdCryptosslX509Crl(XrdCryptoX509 *cacert);
   virtual ~XrdCryptosslX509Crl();

   // Status
   bool IsValid() { return (crl != 0); }

   // Access underlying data (in opaque form: used in chains)
   XrdCryptoX509Crldata Opaque() { return (XrdCryptoX509Crldata)crl; }

   // Dump information
   void Dump();
   const char *ParentFile() { return (const char *)(srcfile.c_str()); }

   // Validity interval
   time_t LastUpdate();  // time when last updated
   time_t NextUpdate();  // time foreseen for next update

   // Issuer of top certificate
   const char *Issuer();
   const char *IssuerHash(int);   // hash 

   // Chec certificate revocation
   bool IsRevoked(int serialnumber, int when = 0);
   bool IsRevoked(const char *sernum, int when = 0);

   // Verify signature
   bool Verify(XrdCryptoX509 *ref);

   // Dump CRL object to a file.
   bool ToFile(FILE *fh);

   //Returns true if the CRL certificate has critical extension, false otherwise
   bool hasCriticalExtension();

private:
   X509_CRL    *crl{nullptr};   // The CRL object
   time_t       lastupdate{-1}; // time of last update
   time_t       nextupdate{-1}; // time of next update
   XrdOucString issuer;         // issuer name;
   XrdOucString issuerhash;     // hash of issuer name (default algorithm);
   XrdOucString issueroldhash;  // hash of issuer name (md5 algorithm);
   XrdOucString srcfile;        // source file name, if any;
   XrdOucString crluri;         // URI from where to get the CRL file, if any;

   int          nrevoked{0};    // Number of certificates revoked
   XrdSutCache  cache;          // cached infor about revoked certificates

   int GetFileType(const char *crlfn); //Determine file type
   int LoadCache();         // Load the cache
   int Init(const char *crlf); // Init from file
   int Init(FILE *fc, const char *crlf); // Init from file handle
   int InitFromURI(const char *uri, const char *hash); // Init from URI
};

#endif
