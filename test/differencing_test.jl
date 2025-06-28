using Pythia
using Test

y = [10, 12, 14, 13, 15, 17, 16,   # Week 1
     13, 15, 17, 16, 18, 20, 19,   # Week 2
     16, 18, 20, 19, 21, 23, 22]   # Week 3

l = difference_series_(y; s=7, alpha=0.05)

# Test that the differenced series is (almost) zero
@test all(isapprox.(l, 0.0; atol=1e-8))
