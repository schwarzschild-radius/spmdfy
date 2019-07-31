#include <cuda_runtime.h>
#include <iostream>
#include <vector>

struct A{
    int a;
    A(int a) : a(a){}
};

int i = 0;
double d = 0;
char c = 'a';
float f = 0.0f;
unsigned int ui = 0;
unsigned char uc = 'b';
bool b = false;
int64_t i32 = 10;
const int ca = 10;
A sb(10);