# obtain parameters using the method of moments
function mom_lognormal(xbar::Float64, s²::Float64)
    μ = log(xbar / sqrt(s²/ xbar^2 + 1)); 
    σ = sqrt(log(s²/ xbar^2 + 1)); 

    return μ, σ
end; 