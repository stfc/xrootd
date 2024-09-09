
/*
 * XrdThrottleManager
 *
 * This class provides an implementation of a throttle manager.
 * The throttled manager purposely pause if the bandwidth, IOPS
 * rate, or number of outstanding IO requests  is sustained above 
 * a certain level.
 *
 * The XrdThrottleManager is user-aware and provides fairshare.
 *
 * This works by having a separate thread periodically refilling
 * each user's shares.
 *
 * Note that we do not actually keep close track of users, but rather
 * put them into a hash.  This way, we can pretend there's a constant
 * number of users and use a lock-free algorithm.
 */

#ifndef __XrdThrottleManager_hh_
#define __XrdThrottleManager_hh_

#ifdef __GNUC__
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#else
#define likely(x)       x
#define unlikely(x)     x
#endif

#include <string>
#include <vector>
#include <ctime>
#include <mutex>
#include <unordered_map>
#include <memory>

#include "XrdSys/XrdSysPthread.hh"

class XrdSysError;
class XrdOucTrace;
class XrdThrottleTimer;
class XrdXrootdGStream;

class XrdThrottleManager
{

friend class XrdThrottleTimer;

public:

void        Init();

bool        OpenFile(const std::string &entity, std::string &open_error_message);
bool        CloseFile(const std::string &entity);

void        Apply(int reqsize, int reqops, int uid);

bool        IsThrottling() {return (m_ops_per_second > 0) || (m_bytes_per_second > 0);}

void        SetThrottles(float reqbyterate, float reqoprate, int concurrency, float interval_length)
            {m_interval_length_seconds = interval_length; m_bytes_per_second = reqbyterate;
             m_ops_per_second = reqoprate; m_concurrency_limit = concurrency;}

void        SetLoadShed(std::string &hostname, unsigned port, unsigned frequency)
            {m_loadshed_host = hostname; m_loadshed_port = port; m_loadshed_frequency = frequency;}

void        SetMaxOpen(unsigned long max_open) {m_max_open = max_open;}

void        SetMaxConns(unsigned long max_conns) {m_max_conns = max_conns;}

void        SetMonitor(XrdXrootdGStream *gstream) {m_gstream = gstream;}

//int         Stats(char *buff, int blen, int do_sync=0) {return m_pool.Stats(buff, blen, do_sync);}

static
int         GetUid(const char *username);

XrdThrottleTimer StartIOTimer();

void        PrepLoadShed(const char *opaque, std::string &lsOpaque);

bool        CheckLoadShed(const std::string &opaque);

void        PerformLoadShed(const std::string &opaque, std::string &host, unsigned &port);

            XrdThrottleManager(XrdSysError *lP, XrdOucTrace *tP);

           ~XrdThrottleManager() {} // The buffmanager is never deleted

protected:

void        StopIOTimer(struct timespec);

private:

void        Recompute();

void        RecomputeInternal();

static
void *      RecomputeBootstrap(void *pp);

int         WaitForShares();

void        GetShares(int &shares, int &request);

void        StealShares(int uid, int &reqsize, int &reqops);

XrdOucTrace * m_trace;
XrdSysError * m_log;

XrdSysCondVar m_compute_var;

// Controls for the various rates.
float       m_interval_length_seconds;
float       m_bytes_per_second;
float       m_ops_per_second;
int         m_concurrency_limit;

// Maintain the shares
static const
int         m_max_users;
std::vector<int> m_primary_bytes_shares;
std::vector<int> m_secondary_bytes_shares;
std::vector<int> m_primary_ops_shares;
std::vector<int> m_secondary_ops_shares;
int         m_last_round_allocation;

// Active IO counter
int         m_io_active;
struct timespec m_io_wait;
unsigned    m_io_total{0};
// Stable IO counters - must hold m_compute_var lock when reading/writing;
int m_stable_io_active;
int m_stable_io_total{0}; // It would take ~3 years to overflow a 32-bit unsigned integer at 100Hz of IO operations.
struct timespec m_stable_io_wait;

// Load shed details
std::string m_loadshed_host;
unsigned m_loadshed_port;
unsigned m_loadshed_frequency;
int m_loadshed_limit_hit;

// Maximum number of open files
unsigned long m_max_open{0};
unsigned long m_max_conns{0};
std::unordered_map<std::string, unsigned long> m_file_counters;
std::unordered_map<std::string, unsigned long> m_conn_counters;
std::unordered_map<std::string, std::unique_ptr<std::unordered_map<pid_t, unsigned long>>> m_active_conns;
std::mutex m_file_mutex;

// Monitoring handle, if configured
XrdXrootdGStream* m_gstream{nullptr};

static const char *TraceID;

};

class XrdThrottleTimer
{

friend class XrdThrottleManager;

public:

void StopTimer()
{
   struct timespec end_timer = {0, 0};
#if defined(__linux__) || defined(__APPLE__) || defined(__GNU__) || (defined(__FreeBSD_kernel__) && defined(__GLIBC__))
   int retval = clock_gettime(clock_id, &end_timer);
#else
   int retval = -1;
#endif
   if (likely(retval == 0))
   {
      end_timer.tv_sec -= m_timer.tv_sec;
      end_timer.tv_nsec -= m_timer.tv_nsec;
      if (end_timer.tv_nsec < 0)
      {
         end_timer.tv_sec--;
         end_timer.tv_nsec += 1000000000;
      }
   }
   if (m_timer.tv_nsec != -1)
   {
      m_manager.StopIOTimer(end_timer);
   }
   m_timer.tv_sec = 0;
   m_timer.tv_nsec = -1;
}

~XrdThrottleTimer()
{
   if (!((m_timer.tv_sec == 0) && (m_timer.tv_nsec == -1)))
   {
      StopTimer();
   }
}

protected:

XrdThrottleTimer(XrdThrottleManager & manager) :
   m_manager(manager)
{
#if defined(__linux__) || defined(__APPLE__) || defined(__GNU__) || (defined(__FreeBSD_kernel__) && defined(__GLIBC__))
   int retval = clock_gettime(clock_id, &m_timer);
#else
   int retval = -1;
#endif
   if (unlikely(retval == -1))
   {
      m_timer.tv_sec = 0;
      m_timer.tv_nsec = 0;
   }
}

private:
XrdThrottleManager &m_manager;
struct timespec m_timer;

static clockid_t clock_id;
};

#endif
