//------------------------------------------------------------------------------
// This file is part of XrdTpcTPC
//
// Copyright (c) 2023 by European Organization for Nuclear Research (CERN)
// Author: Cedric Caffy <ccaffy@cern.ch>
// File Date: Oct 2023
//------------------------------------------------------------------------------
// XRootD is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// XRootD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with XRootD.  If not, see <http://www.gnu.org/licenses/>.
//------------------------------------------------------------------------------


#include <sstream>
#include "XrdTpcPMarkManager.hh"

namespace XrdTpc
{
PMarkManager::SocketInfo::SocketInfo(int fd, const struct sockaddr * sockP) {
  netAddr.Set(sockP,fd);
  client.addrInfo = static_cast<XrdNetAddrInfo*>(&netAddr);
}

PMarkManager::PMarkManager(XrdNetPMark *pmark) : mPmark(pmark), mTransferWillStart(false) {}

void PMarkManager::addFd(int fd, const struct sockaddr * sockP) {
  if(mPmark && mTransferWillStart && mReq->mSciTag >= 0) {
    // The transfer will start and the packet marking has been configured, this socket must be registered for future packet marking
    mSocketInfos.emplace(fd, sockP);
  }
}

void PMarkManager::startTransfer(XrdHttpExtReq * req) {
  mReq = req;
  mTransferWillStart = true;
}

void PMarkManager::beginPMarks() {
  if(mSocketInfos.empty()) {
    return;
  }

  if(mPmarkHandles.empty()) {
    // Create the first pmark handle
    std::stringstream ss;
    ss << "scitag.flow=" << mReq->mSciTag;
    SocketInfo & sockInfo = mSocketInfos.front();
    auto pmark = mPmark->Begin(sockInfo.client, mReq->resource.c_str(), ss.str().c_str(), "http-tpc");
    if(!pmark) {
      return;
    }
    mPmarkHandles.emplace(sockInfo.client.addrInfo->SockFD(),std::unique_ptr<XrdNetPMark::Handle>(pmark));
    mSocketInfos.pop();
  }

  auto pmarkHandleItor = mPmarkHandles.begin();
  while(!mSocketInfos.empty()) {
    SocketInfo & sockInfo = mSocketInfos.front();
    auto pmark = mPmark->Begin(*sockInfo.client.addrInfo, *(pmarkHandleItor->second), nullptr);
    if (!pmark) {
      // The packet marking handle could not be created from the first handle, let's retry next time
      break;
    }

    int fd = sockInfo.client.addrInfo->SockFD();
    mPmarkHandles.emplace(fd, std::move(std::unique_ptr<XrdNetPMark::Handle>(pmark)));
    mSocketInfos.pop();
  }
}

void PMarkManager::endPmark(int fd) {
  // We need to delete the PMark handle associated to the fd passed in parameter
  // we just look for it and reset the unique_ptr to nullptr to trigger the PMark handle deletion
  mPmarkHandles.erase(fd);
}
} // namespace XrdTpc
