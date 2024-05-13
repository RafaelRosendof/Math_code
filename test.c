/* solve the Knight's tour using Warnsdorff's rule (always moving
   to the square having the fewest onward moves) */

#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

int board[8][8];

typedef struct{
  int x;
  int y;
  }MOVE;

MOVE move[]={
  { 1, 2},
  { 2, 1},
  { 2,-1},
  { 1,-2},
  {-1,-2},
  {-2,-1},
  {-2, 1},
  {-1, 2}};

int MoveOK(int x,int y)
  {
  if(x<0)
    return(0);
  if(x>7)
    return(0);
  if(y<0)
    return(0);
  if(y>7)
    return(0);
  if(board[y][x])
    return(0);
  return(1);
  }

int Moves(int x,int y)
  {
  int i,n;
  n=0;
  for(i=0;i<8;i++)
    n+=MoveOK(x+move[i].x,y+move[i].y);
  return(n);
  }

void ListBoard()
  {
  int i,j;
  for(i=7;i>=0;i--)
    {
    for(j=0;j<8;j+=2)
      printf("°°°°°°ÛÛÛÛÛÛ");
    printf("\n");
    for(j=0;j<8;j+=2)
      printf("°°%02i°°ÛÛ%02iÛÛ",board[i][j],board[i][j+1]);
    printf("\n");
    for(j=0;j<8;j+=2)
      printf("°°°°°°ÛÛÛÛÛÛ");
    printf("\n");
    i--;
    for(j=0;j<8;j+=2)
      printf("ÛÛÛÛÛÛ°°°°°°");
    printf("\n");
    for(j=0;j<8;j+=2)
      printf("ÛÛ%02iÛÛ°°%02i°°",board[i][j],board[i][j+1]);
    printf("\n");
    for(j=0;j<8;j+=2)
      printf("ÛÛÛÛÛÛ°°°°°°");
    printf("\n");
    }
  }

int Tour(int x1,int y1)
  {
  int i,m=0,n1,n2,x2,x3,y2,y3;
  memset(board,0,sizeof(board));
  board[y1][x1]=1;
  while(1)
    {
    n1=9;
    for(i=0;i<8;i++)
      {
      x2=x1+move[i].x;
      y2=y1+move[i].y;
      if(!MoveOK(x2,y2))
        continue;
      if(m==62)
        {
        x3=x2;
        y3=y2;
        break;
        }
      n2=Moves(x2,y2);
      board[y2][x2]=0;
      if(n2<1||n2>=n1)
        continue;
      x3=x2;
      y3=y2;
      n1=n2;
      }
    if(n1==9&&m!=62)
      return(0);
    m++;
    x1=x3;
    y1=y3;
    board[y3][x3]=m+1;
    if(m==63)
      break;
    }
  return(1);
  }

int main(int argc,char**argv,char**envp)
  {
  char bufr[80];
  int x1,y1;
  do{
    printf("enter x,y for first square (1,1) to (8,8) ");
    gets(bufr);
    if(sscanf(bufr,"%i%*[ ,\t]%i",&x1,&y1)!=2)
      break;
    if(x1<1||x1>8)
      {
      fprintf(stderr,"starting column = %i is outside range 0 to 8\n",x1);
      continue;
      }
    x1--;
    if(y1<1||y1>8)
      {
      fprintf(stderr,"starting row = %i is outside range 0 to 8\n",y1);
      continue;
      }
    y1--;
    if(!Tour(x1,y1))
      {
      fprintf(stderr,"knight's tour failed - no solution found\n");
      continue;
      }
    ListBoard();
    }while(1);
  return(0);
  }