using Test

@testset "Diebold-Mariano Tests" begin
    actual = [100.0, 102.0, 104.0, 108.0]
    pred1  = [98.0, 101.0, 103.0, 109.0]
    pred2  = [97.0, 100.0, 105.0, 110.0]

    # MSE
    result_mse = dm_test(actual, pred1, pred2, h=1, crit="MSE")
    @test isapprox(result_mse.DM, -2.6679; atol=1e-3)
    @test isapprox(result_mse.p_value, 0.07583; atol=1e-4)

    # MAD
    result_mad = dm_test(actual, pred1, pred2, h=1, crit="MAD")
    @test isapprox(result_mad.DM, -3.0; atol=1e-3)
    @test isapprox(result_mad.p_value, 0.05767; atol=1e-4)

    # MSE with h = 2
    result_mse_h2 = dm_test(actual, pred1, pred2, h=2, crit="MSE")
    @test isapprox(result_mse_h2.DM, -2.0196; atol=1e-3)
    @test isapprox(result_mse_h2.p_value, 0.1367; atol=1e-4)
end
