# SPECFEM3D FK injection: incident SV bottom-vector indexing bug

**Component:** `src/specfem3D/couple_with_injection.f90` (the `FK` subroutine).
**Impact:** The incident **SV** plane-wave field computed by the FK injection
technique does not satisfy the free-surface zero-traction condition, so the
injected SV wavefield is incorrect. Incident **P** is unaffected.

## Summary

The FK solver builds the half-space mode-amplitude vector `bot_vec` from the two
coefficients returned by the free-surface 2×2 solve. That solve is written **once**
for both P and SV and uses **columns 1 and 3** of `N_mat` (0-based) — i.e. the
down-going SV (col 1) and down-going P (col 3) reflected modes. The P branch places
the coefficients accordingly, but the SV branch places the first coefficient in
**column 2** (the incident P-up slot) and leaves column 1 (reflected SV-down) zero.

## Evidence

Mode/column convention from the `E_mat` (eigenmatrix) construction:
`col 0 = SV-up`, `col 1 = SV-down`, `col 2 = P-up`, `col 3 = P-down` (0-based).

The shared 2×2 solve (0-based column indices 1 and 3):

```fortran
! a = N_mat(3,2), b = N_mat(3,4), c = N_mat(4,2), d = N_mat(4,4)   (1-based)
! -> columns 1 and 3 (0-based) of the traction rows 2 and 3 (0-based)
```

Bottom-vector construction (`couple_with_injection.f90`, ~lines 1188-1196, 1-based):

```fortran
bot_vec(:) = 0
if (kpsv == 1) then          ! P
  bot_vec(2) = coeff(1,ii)   ! 0-based [1] = reflected SV-down   (CORRECT)
  bot_vec(3) = C_3           ! 0-based [2] = incident P-up
  bot_vec(4) = coeff(2,ii)   ! 0-based [3] = reflected P-down    (CORRECT)
else                         ! SV
  bot_vec(1) = C_1           ! 0-based [0] = incident SV-up
  bot_vec(3) = coeff(1,ii)   ! 0-based [2] = incident P-up slot  (WRONG: should be [1])
  bot_vec(4) = coeff(2,ii)   ! 0-based [3] = reflected P-down
end if
```

For SV, `coeff(1,ii)` is the reflected **SV-down** amplitude (column 1), but it is
written into `bot_vec(3)` (0-based index 2, the incident P-up slot), while
`bot_vec(2)` (0-based index 1, reflected SV-down) is left zero.

## Why it is wrong

The free-surface condition is zero traction at the surface: with
`y = N_mat · bot_vec`, the traction rows `y(2)` and `y(3)` (0-based) must vanish.
The solve enforces, for both P and SV,

```
N(2,0)·known + N(2,1)·coeff0 + N(2,3)·coeff1 = 0
N(3,0)·known + N(3,1)·coeff0 + N(3,3)·coeff1 = 0
```

i.e. `coeff0` multiplies **column 1** and `coeff1` multiplies **column 3**. The P
branch places `coeff0` at index 1 and the condition holds. The SV branch places
`coeff0` at index 2, so the actual traction becomes

```
y(2) = N(2,0)·C_1 + N(2,2)·coeff0 + N(2,3)·coeff1  ≠ 0
```

(unless `N(2,1) == N(2,2)`, which is not generally true), so the SV free-surface
traction does not vanish.

## Fix

In the SV branch, place `coeff(1,ii)` at the reflected SV-down slot (0-based
column 1), matching the P branch and the columns used by the solve:

```fortran
else                         ! SV
  bot_vec(1) = C_1           ! 0-based [0] = incident SV-up
  bot_vec(2) = coeff(1,ii)   ! 0-based [1] = reflected SV-down  (FIXED)
  bot_vec(4) = coeff(2,ii)   ! 0-based [3] = reflected P-down
end if
```

## Verification

In the SPECFEM++ native FK port we apply the fixed placement and assert the
free-surface zero-traction condition for a single elastic half-space for **both** P
and SV incidence (unit test `HalfspaceSolve.FreeSurfaceTractionVanishes{P,SV}`):
with the fix, `|y(2)|` and `|y(3)|` cancel to machine precision relative to the
traction-row term magnitudes; with the original indexing, the SV case fails.

## Downstream notes

- The Westervelt `tools/wsem/_internal` C++ FK port inherits this bug verbatim
  (`fk_core.cpp`, the `kpsv != 1` branch of the bottom-vector construction).
- Any FK-injection validation that only exercised **P** incidence would not have
  caught this.
