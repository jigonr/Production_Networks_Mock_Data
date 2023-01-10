# packages
using Random; 
using Distributions; 
using Statistics; 
using SplitApplyCombine; 

include("stats.jl")

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
Nᶠ = Int64(16_384); 
Nₒᵇ = Int64(0.25 * Nᶠ); 
Nₒˢ = Int64(0.10 * Nᶠ); 
Nˢᵇ = Int64(0.65 * Nᶠ); 

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
bⱼ = floor.(rand(dₛᵤₚ, Nᶠ)) .+ 1; 
sample = trues(Nₒᵇ); 
append!(sample, rand(Nₒˢ + Nˢᵇ) .>= Nₒˢ/ (Nₒˢ + Nˢᵇ)); 
bⱼ .*= sample; 

# extensive margin
pᵢⱼ = 1 .- (1 .- sᵢ) .^ bⱼ'; 

# intensive margin
xᵢⱼ = reshape(combinedims(rand.(Binomial.(bⱼ', sᵢ), 1)), (Nᶠ, Nᶠ)); 
