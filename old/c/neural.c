
#include <math.h>

typedef struct {
    int length;
    float *data;
} vec_t;

typedef struct {
    int w, h;
    float *data;
} mat_t;

void freeVec(vec_t* x)
{
    if(x->data != NULL) {
        free(x->data);
        x->data = NULL;
        x->length = 0;
    }
}

void freeMat(mat_t* m)
{
    if(m->data != NULL) {
        free(m->data);
        m->data = NULL;
        m->w = 0;
        m->h = 0;
    }
}

vec_t newVec(int length)
{
    vec_t v;
    v.length = length;
    v.data = (float*)malloc(sizeof(float)*length);
    for(int i = 0; i < length; i++) v.data[i] = 0.0;
    return v;
}

mat_t newMat(int w, int h)
{
    mat_t m;
    m.w = w;
    m.h = h;
    m.data = (float*)malloc(sizeof(float)*w*h);
    for(int i = 0; i < w*h; i++) m.data[i] = 0.0;
    return m;
}

void matMultiply(mat_t m, vec_t x, vec_t* v)
{
    if(m.w != x.length) 
        return;
    float d;
    for(int i = 0; i < m.h; i++) {
        d = 0.0;
        for(int j = 0; j < m.w; i++) {
            d += m.data[i*m.w+j];
        }
        v.data[i] = d;
    }

    return v;
}

float dot(vec_t v1, vec_t v2)
{
    if(v1.length != v2.length) return 0.0;
    float d = 0.0;
    for(int i = 0; i < v1.length; i++)
        d += v1[i]*v2[i];
    return d;
}

typedef struct {
    int inputLength, hiddenLength, outputLength;
    mat_t hh, hy, xh;
    vec_t h;
} RNN;

RNN newRNN(int inLen, int hiddenLen, int outLen) {
    RNN rnn;
    rnn.inputLength = inLen;
    rnn.hiddenLength = hiddenLen;
    rnn.outputLength = outLen;
    rnn.hh = newMat(hiddenLen,hiddenLen);
    rnn.hy = newMat(hiddenLen,outLen);
    rnn.hx = newMat(inLen,hiddenLen);
    rnn.h = newVec(hiddenLen);
}

void step(RNN rnn, vec_t x, vec_t* y)
{
    vec_t a, b;
    a = newVec(rnn.h.length);
    b = newVec(rnn.h.length);
    matMultiply(rnn.hh, rnn.h, &a);
    matMultiply(rnn.xh, x, &b);
    for(int i = 0; i < rnn.h.length; i++)
        rnn.h.data[i] = tanh(a[i]+b[i]);
    matMultiply(rnn.hy, rnn.h, y);
    freeVec(a);
    freeVec(b);
}

