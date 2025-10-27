//------------------------------------------------------------------------------
// Copyright (c) 2012-2013 by European Organization for Nuclear Research (CERN)
// Author: Justin Salmon <jsalmon@cern.ch>
//------------------------------------------------------------------------------
// This file is part of the XRootD software suite.
//
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
//
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//------------------------------------------------------------------------------

#include "PyXRootD.hh"
#include "PyXRootDFileSystem.hh"
#include "PyXRootDFile.hh"
#include "PyXRootDCopyProcess.hh"
#include "PyXRootDURL.hh"
#include "PyXRootDFinalize.hh"
#include "PyXRootDEnv.hh"

namespace PyXRootD
{
  // Global module object
  PyObject* ClientModule;

  PyDoc_STRVAR(client_module_doc, "XRootD Client extension module");

  //----------------------------------------------------------------------------
  //! Visible module-level method declarations
  //----------------------------------------------------------------------------
  static PyMethodDef module_methods[] =
    {
      // The finalization routine used in atexit handler.
      { "__XrdCl_Stop_Threads", __XrdCl_Stop_Threads, METH_NOARGS,  "Stop XrdCl threads." },
      // Ths XRootD Env
      { "EnvPutString_cpp",     EnvPutString_cpp,     METH_VARARGS, "Puts a string into XrdCl environment." },
      { "EnvGetString_cpp",     EnvGetString_cpp,     METH_VARARGS, "Gets a string from XrdCl environment." },
      { "EnvPutInt_cpp",        EnvPutInt_cpp,        METH_VARARGS, "Puts an int into XrdCl environment." },
      { "EnvGetInt_cpp",        EnvGetInt_cpp,        METH_VARARGS, "Gets an int from XrdCl environment." },
      { "XrdVersion_cpp",       XrdVersion_cpp,       METH_VARARGS, "Get the XRootD client version." },
      { "EnvGetDefault_cpp",    EnvGetDefault_cpp,    METH_VARARGS, "Get default values from XrdCl environment" },
      { NULL, NULL, 0, NULL }
    };

  //----------------------------------------------------------------------------
  //! Module properties
  //----------------------------------------------------------------------------
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "client",              /* m_name */
    client_module_doc,     /* m_doc */
    -1,                    /* m_size */
    module_methods,        /* m_methods */
    NULL,                  /* m_reload */
    NULL,                  /* m_traverse */
    NULL,                  /* m_clear */
    NULL,                  /* m_free */
  };

  //----------------------------------------------------------------------------
  //! Module initialization function
  //----------------------------------------------------------------------------
  PyMODINIT_FUNC PyInit_client( void )
  {
    FileSystemType.tp_new = PyType_GenericNew;
    if ( PyType_Ready( &FileSystemType ) < 0 ) {
      return NULL;
    }
    Py_INCREF( &FileSystemType );

    FileType.tp_new = PyType_GenericNew;
    if ( PyType_Ready( &FileType ) < 0 ) {
      return NULL;
    }
    Py_INCREF( &FileType );

    URLType.tp_new = PyType_GenericNew;
    if ( PyType_Ready( &URLType ) < 0 ) {
      return NULL;
    }
    Py_INCREF( &URLType );

    CopyProcessType.tp_new = PyType_GenericNew;
    if ( PyType_Ready( &CopyProcessType ) < 0 ) {
      return NULL;
    }
    Py_INCREF( &CopyProcessType );

    ClientModule = PyModule_Create(&moduledef);

    if (ClientModule == NULL) {
      return NULL;
    }

    PyModule_AddObject( ClientModule, "FileSystem", (PyObject *) &FileSystemType );
    PyModule_AddObject( ClientModule, "File", (PyObject *) &FileType );
    PyModule_AddObject( ClientModule, "URL", (PyObject *) &URLType );
    PyModule_AddObject( ClientModule, "CopyProcess", (PyObject *) &CopyProcessType );

    return ClientModule;
  }
}
