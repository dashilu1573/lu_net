%module net

%{
/* Includes the header in the wrapper code */
#include "../../include/net.h"
%}

/* Parse the header file to generate wrappers */
%include "../../include/net.h"