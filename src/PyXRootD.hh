//------------------------------------------------------------------------------
// Copyright (c) 2012-2013 by European Organization for Nuclear Research (CERN)
// Author: Justin Salmon <jsalmon@cern.ch>
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

#ifndef PYXROOTD_HH_
#define PYXROOTD_HH_

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#include <Python.h>
#include <string>
#include <iostream>
#include "structmember.h"

#if PY_MINOR_VERSION < 5
#define PyUnicode_FromString PyString_FromString
#endif

#define async( func )    \
  Py_BEGIN_ALLOW_THREADS \
  func;                  \
  Py_END_ALLOW_THREADS   \

#endif /* PYXROOTD_HH_ */
