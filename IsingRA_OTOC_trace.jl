using LinearAlgebra, KrylovKit, Random, JLD2

function apply_sigma_z!(ψ_out::Vector{ComplexF32}, ψ_in::Vector{ComplexF32}, site::Int)
    @inbounds for b in 0:(length(ψ_in)-1)
        sign = ((b >> (site-1)) & 1) == 1 ? -1.0f0 : 1.0f0
        ψ_out[b+1] = sign * ψ_in[b+1]
    end
    return ψ_out
end
apply_sigma_z(ψ::Vector{ComplexF32}, site::Int) = apply_sigma_z!(similar(ψ), ψ, site)

function apply_sigma_x!(ψ_out::Vector{ComplexF32}, ψ_in::Vector{ComplexF32}, site::Int)
    @inbounds for b in 0:(length(ψ_in)-1)
        b_flipped = b ⊻ (1 << (site - 1))
        ψ_out[b+1] = ψ_in[b_flipped+1]
    end
    return ψ_out
end
apply_sigma_x(ψ::Vector{ComplexF32}, site::Int) = apply_sigma_x!(similar(ψ), ψ, site)

function apply_H!(ψ_out::Vector{ComplexF32}, ψ_in::Vector{ComplexF32}, N::Int, J::Float32, λ::Float32, h::Float32, hA::Float32, g::Float32, gA::Float32)
    dim = length(ψ_in)
    
    @inbounds for b in 0:(dim-1)
        diag_val = 0.0f0
        
        # Ancilla Z
        sz_ancilla = ((b >> N) & 1) == 1 ? -0.5f0 : 0.5f0
        diag_val -= gA * sz_ancilla
        
        # Chain ZZ
        for j in 1:(N-1)
            sz_j   = ((b >> (j-1)) & 1) == 1 ? -0.5f0 : 0.5f0
            sz_jp1 = ((b >> j) & 1) == 1 ? -0.5f0 : 0.5f0
            diag_val += J * sz_j * sz_jp1
        end
        
        # λ coupling and Chain Z
        for j in 1:N
            sz_j = ((b >> (j-1)) & 1) == 1 ? -0.5f0 : 0.5f0
            diag_val += λ * sz_j * sz_ancilla - g * sz_j
        end
        
        # Start accumulating output with the diagonal term
        out_val = diag_val * ψ_in[b+1]
        
        # X fields for Chain
        for j in 1:N
            b_flipped = b ⊻ (1 << (j-1))
            out_val -= h * 0.5f0 * ψ_in[b_flipped+1]
        end
        
        # X field for Ancilla
        b_flipped = b ⊻ (1 << N)
        out_val -= hA * 0.5f0 * ψ_in[b_flipped+1]
        
        ψ_out[b+1] = out_val
    end
    return ψ_out
end

function simulate_otoc_trace(j::Int, k::Int, N::Int, tmax::Real, dt::Real, λarr::Vector{Float64}; J::Real = 1.0, h::Real=1.0, hA::Real=1.0, g::Real=0.0, gA::Real=0.0)
    @assert j ≤ N+1 "j must be ≤ (N + 1)"
    @assert k ≤ N+1 "k must be ≤ (N + 1)"

    times = Float32(dt):Float32(dt):Float32(tmax)
    dim = 1 << (N+1)
    
    original_blas_threads = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)
    
    ψ_inf_global = randn(ComplexF32, dim)
    ψ_inf_global ./= norm(ψ_inf_global) 
    
    Threads.@threads for l in 1:length(λarr)
        λ = Float32(λarr[l])
        
        # Define the matrix-free action for KrylovKit
        H_action = ψ -> apply_H!(similar(ψ), ψ, N, Float32(J), λ, Float32(h), Float32(hA), Float32(g), Float32(gA))
        
        ψ_0_t = copy(ψ_inf_global)
        ψ_V_t = apply_sigma_z(ψ_inf_global, k) 
        
        ψ_W_0_t = similar(ψ_0_t)
        ψ_W_V_t = similar(ψ_V_t)
        
        otoc_matrix = zeros(ComplexF32, length(times))
        for (i, t) in enumerate(times)
            # Forward evolution by dt
            ψ_0_t, _ = exponentiate(H_action, -1im * Float32(dt), ψ_0_t; ishermitian=true, krylovdim=30)
            ψ_V_t, _ = exponentiate(H_action, -1im * Float32(dt), ψ_V_t; ishermitian=true, krylovdim=30)

            # Apply W in-place
            apply_sigma_z!(ψ_W_0_t, ψ_0_t, j)
            apply_sigma_z!(ψ_W_V_t, ψ_V_t, j)

            back_0, _ = exponentiate(H_action, 1im * t, ψ_W_0_t; ishermitian=true, krylovdim=30)
            back_V, _ = exponentiate(H_action, 1im * t, ψ_W_V_t; ishermitian=true, krylovdim=30)
            
            # Measure
            apply_sigma_z!(ψ_W_V_t, back_V, k)
            otoc_matrix[i] = dot(back_0, ψ_W_V_t)
        end
        
        save_object("../../data/OTOC_data/IsingRA_Czz_trace_N=$(N)_r=$(k)_lambda=$(round(λ,digits=2))_J=1.0_h=1.05_g=0.45.jld2", (tarr, otoc_matrix))

    end
    
    LinearAlgebra.BLAS.set_num_threads(original_blas_threads)
    return
end

N = 10;
j = 1;

h = 1.05;
g = 0.45;

k = N - 1; ## ancilla = N+1

λarr = collect(10 .^ range(-1,1.5,length=100));
tmax = 30.; dt = 0.05; tarr = collect(dt:dt:tmax)

simulate_otoc_trace(j, k, N, tmax, dt, λarr; J = 1., h=h, hA=h, g=g, gA=g);
