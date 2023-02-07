# packages
using CSV; 
using DataFrames; 
using Distributions; 
using Random; 
using Statistics; 

# obtain parameters using the method of moments
function mom_lognormal(xbar::Float64, s²::Float64)
    μ = log(xbar / sqrt(s²/ xbar^2 + 1)); 
    σ = sqrt(log(s²/ xbar^2 + 1)); 

    return μ, σ
end; 

# set seed
seed = 2023; 
Random.seed!(seed); 

# moments from BCCR
## number of suppliers distribution
xbarₛᵤₚ = 20.88; 
sₛᵤₚ² = 130.59^2; 

## transactions distribution
xbarₜᵣₐₙₛ = 0.03359; 
sₜᵣₐₙₛ² = 0.05853^2; 

## sales distribution
xbarₛₐₗₑₛ = 2.03; 
sₛₐₗₑₛ² = 11.92^2; 

## transactions parameters
μₜᵣₐₙₛ, σₜᵣₐₙₛ = mom_lognormal(xbarₜᵣₐₙₛ, sₜᵣₐₙₛ²); 
μₛₐₗₑₛ, σₛₐₗₑₛ = mom_lognormal(xbarₛₐₗₑₛ, sₛₐₗₑₛ²); 
μₛᵤₚ, σₛᵤₚ = mom_lognormal(xbarₛᵤₚ, sₛᵤₚ²); 

# create distributions
dₜᵣₐₙₛ = LogNormal(μₜᵣₐₙₛ, σₜᵣₐₙₛ); 
dₛₐₗₑₛ = LogNormal(μₛₐₗₑₛ, σₛₐₗₑₛ); 
dₛᵤₚ = LogNormal(μₛᵤₚ, σₛᵤₚ); # three parameter: location 1 by adding 1 after sampling, conditional on buying

# set number of firms
Nᶠ = Int64(32_000); 
Nₒᵇ = floor(Int64, 0.31 * Nᶠ); 
Nₒˢ = floor(Int64, 0.13 * Nᶠ); 
Nˢᵇ = ceil(Int64, 0.56 * Nᶠ); 

# draw sales for only sellers and simultaneous buyers and sellers
## we assume the market share is equal to the intermediate input market share
sᵢ = rand(dₛₐₗₑₛ, Nₒˢ + Nˢᵇ); 

# add only buyers
sᵢ₀ = zeros(Float64, Nₒᵇ); 
append!(sᵢ, sᵢ₀); 
sort!(sᵢ); 

# compute shares sellers
sᵢ .= sᵢ / sum(sᵢ); 

# add only buyers
bⱼ = floor.(Int64, rand(dₛᵤₚ, Nᶠ)) .+ 1; 
sample = trues(Nₒᵇ); 
append!(sample, rand(Nₒˢ + Nˢᵇ) .>= Nₒˢ/ (Nₒˢ + Nˢᵇ)); 
bⱼ .*= sample; 
bⱼ = Array{Int64}(transpose(bⱼ)); 

# extensive margin
# preallocate space
pᵢⱼ = zeros(Float64, Nᶠ, Nᶠ); 

# compute probabilities of linkages
pᵢⱼ .= 1 .- (1 .- sᵢ) .^ bⱼ; 

# intensive margin
# draw number of transactions between firms
@inline function intensive_margin(sᵢ, bⱼ)
    xᵢⱼ = zeros(Int64, length(sᵢ), length(bⱼ)); 
    @views @inbounds for i ∈ eachindex(sᵢ), j ∈ eachindex(bⱼ)
        xᵢⱼ[i, j] = rand(Binomial(bⱼ[j], sᵢ[i]), 1)[1]
    end
    return xᵢⱼ
end; 

xᵢⱼ = intensive_margin(sᵢ, bⱼ); 

# create df to save treansactions
# compute cartesian product with ID combinations
transactions = DataFrame(Iterators.product(1:length(sᵢ), 1:length(bⱼ))); 
rename!(transactions, [:seller, :buyer]); 

# add transactions to df
transactions.trans = vec(xᵢⱼ); 

# drop zero value transactions
filter!(row -> row.trans > 0, transactions); 

# save df
CSV.write("C:/Users/jigon/OneDrive/Escritorio/mock.csv", transactions); 
