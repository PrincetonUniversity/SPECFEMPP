/* setup/config.h.  Generated from config.h.in by configure.  */
/* setup/config.h.in.  Generated from configure.ac by autoheader.  */

/* Define to dummy 'main' function (if any) required to link to the Fortran
   libraries. */
/* #undef FC_DUMMY_MAIN */

/* Define if F77 and FC dummy 'main' functions are identical. */
/* #undef FC_DUMMY_MAIN_EQ_F77 */

/* Define to a macro mangling the given C identifier (in lower and upper
   case), which must not contain underscores, for linking with Fortran. */
#define FC_FUNC(name, NAME) name##_

/* As FC_FUNC, but for C identifiers containing underscores. */
#define FC_FUNC_(name, NAME) name##_

/* Define if emmintrin.h */
/* #undef HAVE_EMMINTRIN */

/* Define if err.h */
#define HAVE_ERR 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the 'vtkCommon' library (-lvtkCommon). */
/* #undef HAVE_LIBVTKCOMMON */

/* Define to 1 if you have the 'vtkDICOMParser' library (-lvtkDICOMParser). */
/* #undef HAVE_LIBVTKDICOMPARSER */

/* Define to 1 if you have the 'vtkexpat' library (-lvtkexpat). */
/* #undef HAVE_LIBVTKEXPAT */

/* Define to 1 if you have the 'vtkFiltering' library (-lvtkFiltering). */
/* #undef HAVE_LIBVTKFILTERING */

/* Define to 1 if you have the 'vtkGenericFiltering' library
   (-lvtkGenericFiltering). */
/* #undef HAVE_LIBVTKGENERICFILTERING */

/* Define to 1 if you have the 'vtkGraphics' library (-lvtkGraphics). */
/* #undef HAVE_LIBVTKGRAPHICS */

/* Define to 1 if you have the 'vtkRendering' library (-lvtkRendering). */
/* #undef HAVE_LIBVTKRENDERING */

/* Define to 1 if you have the 'vtksys' library (-lvtksys). */
/* #undef HAVE_LIBVTKSYS */

/* Define to 1 if you have the 'vtkzlib' library (-lvtkzlib). */
/* #undef HAVE_LIBVTKZLIB */

/* Define if you have POSIX threads libraries and header files. */
/* #undef HAVE_PTHREAD */

/* defined if Scotch is installed */
#define HAVE_SCOTCH 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* define if the VTK library is available */
/* #undef HAVE_VTK */

/* Define if xmmintrin.h */
/* #undef HAVE_XMMINTRIN */

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "see the wiki"

/* Define to the full name of this package. */
#define PACKAGE_NAME "Specfem3D"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "Specfem3D 4.1.1"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "Specfem3D"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "4.1.1"

/* Define to necessary symbol if this constant uses a non-standard name on
   your system. */
/* #undef PTHREAD_CREATE_JOINABLE */

/* Define GIT branch for package source. */
#define SPECFEM3D_GIT_BRANCH "devel"

/* Define date of GIT commit for package source. */
#define SPECFEM3D_GIT_DATE "2024-11-21 11:50:38 +0100"

/* Define GIT hash for package source. */
#define SPECFEM3D_GIT_HASH "725321e6599f90e486c907435840df559ebc0a9c"

/* Define git revision commit for package source. */
#define SPECFEM3D_GIT_REVISION "v4.1.1-100-g725321e6"

/* Set to 0 if source is from GIT, 1 otherwise. */
#define SPECFEM3D_RELEASE_VERSION 0

/* Define SPECFEM3D version */
#define SPECFEM3D_VERSION "4.1.1"

/* Define to 1 if all of the C89 standard headers exist (not just the ones
   required in a freestanding environment). This macro is provided for
   backward compatibility; new code need not use it. */
#define STDC_HEADERS 1

/* Define to 1 if 'lex' declares 'yytext' as a 'char *' by default, not a
   'char[]'. */
/* #undef YYTEXT_POINTER */

/* Define to select optimized file i/o for regional simulations */
/* map fails when output files are > 4GB, which is often the case for GPU
 * simulations */
/* #undef USE_MAP_FUNCTION */
