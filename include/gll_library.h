#ifndef GLL_LIBRARY_H
#define GLL_LIBRARY_H

namespace gll_library {
    float pnleg(const float z, const int n);
    float pnglj(const float z, const int n);
    float pndleg(const float z, const int n);
    float pndglj(const float z, const int n);
    class gll{
        public:
            gll();
            gll(const double alpha, const double beta, const int ngll);
        
        private:
            float alpha, beta;
            int ngll;
    };
}

#endif // GLL_LIBRARY_H