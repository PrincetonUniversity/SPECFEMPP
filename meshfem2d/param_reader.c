/*
 !========================================================================
 !
 !                           S P E C F E M 2 D
 !                           -----------------
 !
 !     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
 !                              CNRS, France
 !                       and Princeton University, USA
 !                 (there are currently many more authors!)
 !                           (c) October 2017
 !
 ! This software is a computer program whose purpose is to solve
 ! the viscoelastic anisotropic or poroelastic wave equation
 ! using a spectral-element method (SEM).
 !
 ! This program is free software; you can redistribute it and/or modify
 ! it under the terms of the GNU General Public License as published by
 ! the Free Software Foundation; either version 3 of the License, or
 ! (at your option) any later version.
 !
 ! This program is distributed in the hope that it will be useful,
 ! but WITHOUT ANY WARRANTY; without even the implied warranty of
 ! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 ! GNU General Public License for more details.
 !
 ! You should have received a copy of the GNU General Public License along
 ! with this program; if not, write to the Free Software Foundation, Inc.,
 ! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 !
 ! The full text of the license is available in file "LICENSE".
 !
 !========================================================================
 */

/*

 by Dennis McRitchie (Princeton University, USA) and others

 January 7, 2010 - par_file parsing
 June 1, 2011 - Updated to support multi-word values; also a few bug fixes.
 ..
 You'll notice that the heart of the parser is a complex regular
 expression that is compiled within the C code, and then used to split
 the lines appropriately. It does all the heavy lifting. I don't know of
 any way to do this in Fortran. I believe that to accomplish this in
 Fortran, you'd have to write a lot of procedural string manipulation
 code, for which Fortran is not very well suited.

 But Fortran-C mixes are pretty common these days, so I would not expect
 any problems on that account. There are no wrapper functions used: just
 the C routine called directly from a Fortran routine. Also, regarding
 the use of C, I assumed this would not be a problem since there are
 already a few C files that make up part of the build.
 ..
*/

#define _GNU_SOURCE
#include "config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <regex.h>

#ifndef LINE_MAX
#define LINE_MAX 255
#endif

/*
 * Mac OS X's gcc does not support strnlen and strndup.
 * So we define them here conditionally, to avoid duplicate definitions
 * on other systems.
 */
#ifdef __APPLE__
size_t strnlen (const char *string, size_t maxlen)
{
  const char *end = memchr (string, '\0', maxlen);
  return end ? (size_t) (end - string) : maxlen;
}

char *strndup (char const *s, size_t n)
{
  size_t len = strnlen (s, n);
  char *new = malloc (len + 1);

  if (new == NULL)
    return NULL;

  new[len] = '\0';
  return memcpy (new, s, len);
}
#endif
/*===============================================================*/

FILE * fid;

void
FC_FUNC_(param_open,PARAM_OPEN)(char * filename, int * length, int * ierr)
{
  char * fncopy;
  char * blank;

  // Trim the file name.
  fncopy = strndup(filename, *length);
  blank = strchr(fncopy, ' ');
  if (blank != NULL) {
    fncopy[blank - fncopy] = '\0';
  }
  if ((fid = fopen(fncopy, "r")) == NULL) {
    // DK DK purposely suppressed this for NUMBER_OF_SIMULTANEOUS_RUNS     printf("Can't open '%s'\n", fncopy);
    *ierr = 1;
    return;
  }
  free(fncopy);
  *ierr = 0;
}

void
FC_FUNC_(param_close,PARAM_CLOSE)()
{
  fclose(fid);
}

void
FC_FUNC_(param_read,PARAM_READ)(char * string_read, int * string_read_len, char * name, int * name_len, int * ierr)
{
  char * namecopy;
  char * blank;
  char * namecopy2;
  int status;
  regex_t compiled_pattern;
  char line[LINE_MAX];
  int regret;
  regmatch_t parameter[3];
  char * keyword;
  char * value;
  size_t value_len;

  // Trim the keyword name we're looking for.
  namecopy = strndup(name, *name_len);
  blank = strchr(namecopy, ' ');
  if (blank != NULL) {
    namecopy[blank - namecopy] = '\0';
  }

  // Then get rid of any dot-terminated prefix.
  namecopy2 = strchr(namecopy, '.');
  if (namecopy2 != NULL) {
    namecopy2 += 1;
  } else {
    namecopy2 = namecopy;
  }

  /* Regular expression for parsing lines from param file.
   ** Good luck reading this regular expression.  Basically, the lines of
   ** the parameter file should be of the form 'parameter = value',
   ** optionally followed by a #-delimited comment.
   ** 'value' can be any number of space- or tab-separated words. Blank
   ** lines, lines containing only white space and lines whose first non-
   ** whitespace character is '#' are ignored.  White space is generally
   ** ignored.  As you will see later in the code, if both parameter and
   ** value are not specified the line is ignored.
   */
  char pattern[] = "^[ \t]*([^# \t]+)[ \t]*=[ \t]*([^# \t]+([ \t]+[^# \t]+)*)";

  // Compile the regular expression.
  status = regcomp(&compiled_pattern, pattern, REG_EXTENDED);
  if (status != 0) {
    printf("regcomp returned error %d\n", status);
  }
  // Position the open file to the beginning.
  if (fseek(fid, 0, SEEK_SET) != 0) {
    printf("Can't seek to beginning of parameter file\n");
    *ierr = 1;
    regfree(&compiled_pattern);
    return;
  }
  // Read every line in the file.
  while (fgets(line, LINE_MAX, fid) != NULL) {
    // Get rid of the ending newline.
    int linelen = strlen(line);
    if (line[linelen-1] == '\n') {
      line[linelen-1] = '\0';
    }
    /* Test if line matches the regular expression pattern, if so
     ** return position of keyword and value */
    regret = regexec(&compiled_pattern, line, 3, parameter, 0);
    // If no match, check the next line.
    if (regret == REG_NOMATCH) {
      continue;
    }
    // If any error, bail out with an error message.
    if (regret != 0) {
      printf("regexec returned error %d\n", regret);
      *ierr = 1;
      regfree(&compiled_pattern);
      return;
    }
    // printf("Line read = %s\n", line);
    // If we have a match, extract the keyword from the line.
    keyword = strndup(line+parameter[1].rm_so, parameter[1].rm_eo-parameter[1].rm_so);

    // If the keyword is not the one we're looking for, check the next line.
    if (strcmp(keyword, namecopy2) != 0) {
      free(keyword);
      continue;
    }
    free(keyword);
    regfree(&compiled_pattern);
    // If it matches, extract the value from the line.
    value = strndup(line+parameter[2].rm_so, parameter[2].rm_eo-parameter[2].rm_so);

    // Clear out the return string with blanks, copy the value into it, and return.
    memset(string_read, ' ', *string_read_len);

    value_len = strlen(value);
    // makes sure there is a character left for NUL termination
    if (value_len > (size_t)*string_read_len - 1) value_len = *string_read_len - 1;

    strncpy(string_read, value, value_len);

    free(value);
    free(namecopy);
    *ierr = 0;
    return;
  }
  // If no keyword matches, set the error flag
  free(namecopy);
  regfree(&compiled_pattern);
  *ierr = 1;
  return;
}


// reads next line in file to read in parameter
void
FC_FUNC_(param_read_nextparam,PARAM_READ_NEXTPARAM)(char * string_read, int * string_read_len, char * name, int * name_len, int * ierr)
{
  char * namecopy;
  char * blank;
  char * namecopy2;
  int status;
  regex_t compiled_pattern;
  char line[LINE_MAX];
  int regret;
  regmatch_t parameter[3];
  char * keyword;
  char * value;
  size_t value_len;

  // Trim the keyword name we're looking for.
  namecopy = strndup(name, *name_len);
  blank = strchr(namecopy, ' ');
  if (blank != NULL) {
   namecopy[blank - namecopy] = '\0';
  }
  // Then get rid of any dot-terminated prefix.
  namecopy2 = strchr(namecopy, '.');
  if (namecopy2 != NULL) {
   namecopy2 += 1;
  } else {
   namecopy2 = namecopy;
  }

  /* Regular expression for parsing lines from param file.
   ** Good luck reading this regular expression.  Basically, the lines of
   ** the parameter file should be of the form 'parameter = value',
   ** optionally followed by a #-delimited comment.
   ** 'value' can be any number of space- or tab-separated words. Blank
   ** lines, lines containing only white space and lines whose first non-
   ** whitespace character is '#' are ignored.  White space is generally
   ** ignored.  As you will see later in the code, if both parameter and
   ** value are not specified the line is ignored.
   */
  char pattern[] = "^[ \t]*([^# \t]+)[ \t]*=[ \t]*([^# \t]+([ \t]+[^# \t]+)*)";

  // Compile the regular expression.
  status = regcomp(&compiled_pattern, pattern, REG_EXTENDED);
  if (status != 0) {
    printf("regcomp returned error %d\n", status);
  }

  // Do not reposition to BOF - use current position where file pointer is now...

  // Read next lines in the file until we have a match.
  while (fgets(line, LINE_MAX, fid) != NULL) {
    // Get rid of the ending newline.
    int linelen = strlen(line);
    if (line[linelen-1] == '\n') {
      line[linelen-1] = '\0';
    }
    /* Test if line matches the regular expression pattern, if so
     ** return position of keyword and value */
    regret = regexec(&compiled_pattern, line, 3, parameter, 0);
    // If no match, check the next line.
    if (regret == REG_NOMATCH) {
      continue;
    }
    // If any error, bail out with an error message.
    if (regret != 0) {
      printf("regexec returned error %d\n", regret);
      *ierr = 1;
      regfree(&compiled_pattern);
      return;
    }
    //printf("Line nextparam read: %s\n", line);

    // If we have a match, extract the keyword from the line.
    keyword = strndup(line+parameter[1].rm_so, parameter[1].rm_eo-parameter[1].rm_so);
    //printf("keyword: %s\n", keyword);

    // If the keyword is not the one we're looking for, return with an error.
    if (strcmp(keyword, namecopy2) != 0) {
      printf("keyword returned wrong parameter %s instead of %s \n", keyword,namecopy2);
      free(keyword);
      *ierr = 1;
      regfree(&compiled_pattern);
      return;
    }
    free(keyword);
    regfree(&compiled_pattern);

    // If it matches, extract the value from the line.
    value = strndup(line+parameter[2].rm_so, parameter[2].rm_eo-parameter[2].rm_so);
    //printf("value: %s\n", value);

    // Clear out the return string with blanks, copy the value into it, and return.
    memset(string_read, ' ', *string_read_len);

    value_len = strlen(value);
    // makes sure there is a character left for NUL termination
    if (value_len > (size_t)*string_read_len - 1) value_len = *string_read_len - 1;

    strncpy(string_read, value, value_len);
    // printf("'%s'='%s'\n", namecopy2, value);

    free(value);
    free(namecopy);
    *ierr = 0;
    return;
  }
  // If no next line matches, print out error and die.
  printf("No match in parameter file for keyword '%s' on next line\n", namecopy);
  free(namecopy);
  regfree(&compiled_pattern);
  *ierr = 1;
  return;
}


// reads next line in file to read in parameters without need of given parameter name
void
FC_FUNC_(param_read_nextline,PARAM_READ_NEXTLINE)(char * string_read, int * string_read_len, int * ierr)
{
  int status;
  regex_t compiled_pattern;
  char line[LINE_MAX];
  int regret;
  regmatch_t parameter[1];
  size_t value_len;

  // Regular expression to skip any comment lines.
  char pattern[] = "^[ \t]*[^#]";

  // Compile the regular expression.
  status = regcomp(&compiled_pattern, pattern, REG_EXTENDED);
  if (status != 0) {
    printf("regcomp returned error %d\n", status);
  }

  // Do not reposition to BOF - use current position where file pointer is now...

  // Read every line in the file.
  while (fgets(line, LINE_MAX, fid) != NULL) {
    // Get rid of the ending newline.
    int linelen = strlen(line);
    if (line[linelen-1] == '\n') {
      line[linelen-1] = '\0';
    }
    // Test if line matches the regular expression pattern
    regret = regexec(&compiled_pattern, line, 1, parameter, 0);
    // If no match (i.e., comment line), check the next line.
    if (regret == REG_NOMATCH) {
      continue;
    }
    // If any error, bail out with an error message.
    if (regret != 0) {
      printf("regexec returned error %d\n", regret);
      *ierr = 1;
      regfree(&compiled_pattern);
      return;
    }
    regfree(&compiled_pattern);
    //printf("Line nextline read: %s\n", line);

    // Clear out the return string with blanks, copy the value into it, and return.
    memset(string_read, ' ', *string_read_len);


    value_len = strlen(line);
    // makes sure there is a character left for NUL termination
    if (value_len > (size_t)*string_read_len - 1) value_len = *string_read_len - 1;

    strncpy(string_read, line, value_len);

    *ierr = 0;
    return;
  }
  // If no next line matches, print out error and die.
  printf("No match in parameter file for next line \n");
  regfree(&compiled_pattern);
  *ierr = 1;
  return;
}
