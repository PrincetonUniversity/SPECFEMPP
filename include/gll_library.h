#ifndef GLL_LIBRARY_H
#define GLL_LIBRARY_H

namespace gll_library {
    double pnleg(const double z, const int n);
    double pnglj(const double z, const int n);
    double pndleg(const double z, const int n);
    double pndglj(const double z, const int n);
    class gll{
        public:
            gll();
            gll(const double alpha, const double beta, const int ngll);
        
        private:
            double alpha, beta;
            int ngll;
    };
}

#endif // GLL_LIBRARY_H