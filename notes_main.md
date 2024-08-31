# CMSC660: Scientific Computing

## Aug 28 2024: Floating Point Arithmetic

C lang
    integer types (NOT standard) (-2^N,2^N-1):
        char    8 
        short   16 
        int     32 
        long    64
Floating types (IEEE standard)
|sign|mantisa|exp| ~ |s|m|e|
m-> $1 + b_i*2^-i$ 

    single precision |s=1|m=23|e=8| ->32 bits/4 bytes
        C -> float 
        Python -> float32 
    double precision |s=1|m=53|e=11| ->64 bits/8 bytes 
        C -> double 
        Python -> float/float64
        Matlab-> default
    overflow -> fatal, flatlines infinity
    underflow -> nonfatal, reseves bits from the mantisa until it is out, flatlines 0
Floating arithmetic (+,-,*,/,sqrt) 
    addition - fl(x op y) === round(x op y) === $(x op y)*(1+\epsilon)$, where $|\epsilon \leq \epsilon_M|$
        reminder: $\epsilon$ is relative error
        def: $\epsilon_M = \text{distance between 1 and neares fp}/2$
            for single $2^-24$
            for double $2^-53$
    subtraction, mult, div, sqrt all likewise

        Example: 
            $fl(x+y+z) = ((x+y)(1+\epsilon_1)+z)(1+\epsilon_2) = ... $
            tip: ignore high order terms e.g. $\epsilon_1\cdot\epsilon_2$

Next time... conditionedness of problems 

