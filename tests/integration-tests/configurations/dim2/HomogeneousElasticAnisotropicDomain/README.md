# Testing anisotropy

Here we are repeating the serial test 1 by defining a homogeneous isotropic
domain using the lame parameters and defining the stiffness matrix manually.

As a result the traces of test 1 and test 9 should be identical!

Note that the  `database.bin` file is not the same as test 1 since it includes
the anisotropic parameters. We generated the `database.bin` file using the
internal `specfempp` mesher since it is the same as the original one at this
point.

The `Par_file` for a homogeneous isotropic medium defines the material like so
```
# elastic: model_number 1 rho Vp Vs 0 0 QKappa Qmu  0 0 0 0 0 0 (for QKappa and Qmu use 9999 to ignore them)
1 1 2700.d0 3000.d0 1732.051d0 0 0 9999 9999 0 0 0 0 0 0
```

We update this description using the anisotropic model parameters and the
conversion
```fortran
mu = rho * vs**2
lambda = rho * (vp**2 - 2 * vs**2)
```

So that `lambda` and `mu` are defined as
```
mu = 2700 * 1732.051**2 = 8'100'001'799.8227 =
lambda = 2700 * (3000**2 - 2 * 1732.051**2) = 8'099'996'400.35
```
Then the relevant stiffness matrix entries are defined
```
c11 = lambda + 2 * mu = 8'100'001'799.8227 + 2 * 8'099'996'400.35 = 24'299'994'600.5
c13 = lambda = 8'099'996'400.35
c15 = 0
c33 = lambda + 2 * mu
c35 = 0
c55 = mu = 8'100'001'799.8227
c12 = lambda
c23 = lambda
c25 = 0
```

```
# anisotropic: model_number 2 rho     c11             c13             c15  c33             c35  c55               c12             c23             c25  0 QKappa Qmu
               1            2 2700.d0 24299994600.5d0 8099996400.35d0 0.d0 24299994600.5d0 0.d0 8100001799.8227d0 8099996400.35d0 8099996400.35d0 0.d0 0 9999   9999
```

Which above is spaced out for clarity but should be entered in the `Par_file`
without the large spaces.

```
1 2 2700.d0 24299994600.5d0 8099996400.35d0 0.d0 24299994600.5d0 0.d0 8100001799.8227d0 8099996400.35d0 8099996400.35d0 0.d0 0 9999 9999
```
