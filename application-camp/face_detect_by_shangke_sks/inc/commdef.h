#pragma once

#include <stdio.h>

#ifndef LOG_INFO
#define ___FILE___ (strstr(__FILE__, SOURCE_DIR) == __FILE__ ? __FILE__ + strlen(SOURCE_DIR) : __FILE__)
#define LOG_INFO(fmt, ...)                                     \
{                                                              \
  printf( "%s::%s():%d ", __FILE__, __FUNCTION__, __LINE__ );  \
  printf(fmt, __VA_ARGS__);                                    \
  printf("\n");                                                \
}
#endif

// 宏操作，若条件为“假”则中断循环
#ifndef BREAK_ON_FALSE
#define BREAK_ON_FALSE( a )          \
  if( 0 == (a) )                     \
  {                                  \
    LOG_INFO( "breaked! (%s)==false", #a );   \
    break;                           \
  }
#endif // ifndef BREAK_ON_FALSE
// 宏操作，若条件为“空”则中断循环
#ifndef BREAK_ON_NULL
#define BREAK_ON_NULL( a )           \
  if( 0 == (a) )                     \
  {                                  \
    LOG_INFO( "breaked! (%s)==NULL", #a );   \
    break;                           \
  }
#endif // ifndef BREAK_ON_NULL
// 宏操作，若条件为“真”则中断循环
#ifndef BREAK_ON_TRUE
#define BREAK_ON_TRUE( a )           \
  if( (a) )                          \
  {                                  \
    LOG_INFO( "breaked! (%s)==TRUE", #a );   \
    break;                           \
  }
#endif // ifndef BREAK_ON_TRUE
#define CONTINUE_ON_FALSE( a )       \
  if( 0 == (a) )                     \
  {                                  \
    LOG_INFO( "continued! (%s)==false", #a );   \
    continue;                        \
  }

#define CONTINUE_ON_NULL( a )        \
  if( 0 == (a) )                     \
  {                                  \
    LOG_INFO( "continued! (%s)==NULL", #a );   \
    continue;                        \
  }
#define CONTINUE_ON_TRUE( a )        \
  if( (a) )                          \
  {                                  \
    LOG_INFO( "continued! (%s)==TRUE", #a );   \
    continue;                        \
  }

#define DEL_MEM( a ) if ( NULL != (a) ) { delete a; a = NULL; }
#define DEL_MEM_ARY( a ) if ( NULL != (a) ) { delete[] a; a = NULL; }

