using KrylovKit
using LinearAlgebra
using JLD2
using Statistics

function apply_H!(v_out::Vector{ComplexF64}, v_in::Vector{ComplexF64}, N, J, h, lambda, hA, gA, g)
    num_states = 1 << (N + 1) # 2^(N+1)
    
    Threads.@threads for state in 0:(num_states-1)
        amp_diag = v_in[state + 1]
        diag_term = 0.0
        out_val = 0.0 * im
        
        # 1. Ancilla terms (Ancilla is bit N)
        ancilla_bit = (state >> N) & 1
        # Branchless calculation for M1 pipeline efficiency
        sz_A = 0.5 - ancilla_bit 
        diag_term += gA * sz_A
        
        # Sx_A term (off-diagonal: look at the state with flipped ancilla bit)
        out_val += hA * 0.5 * v_in[(state ⊻ (1 << N)) + 1]
        
        # 2. Chain terms
        for j in 0:(N-1)
            # Branchless bit decoding
            sz_j = 0.5 - ((state >> j) & 1)
            sz_jp1 = 0.5 - ((state >> ((j + 1) % N)) & 1)
            
            # Diagonal terms: J S^z S^z + lambda S^z S_A^z + g S^z
            diag_term += J * sz_j * sz_jp1 + lambda * sz_j * sz_A + g * sz_j
            
            # Off-diagonal: h S^x (look at the state with flipped bit j)
            out_val += h * 0.5 * v_in[(state ⊻ (1 << j)) + 1]
        end
        
        out_val += diag_term * amp_diag
        v_out[state + 1] = out_val
    end
    return v_out
end

function apply_H_transverse!(v_out::Vector{ComplexF64}, v_in::Vector{ComplexF64}, N, J, h, lambda, hA, gA, g)
    num_states = 1 << (N + 1) # 2^(N+1)
    ancilla_mask = 1 << N     # Precomputed for efficiency
    
    Threads.@threads for state in 0:(num_states-1)
        amp_diag = v_in[state + 1]
        diag_term = 0.0
        out_val = 0.0 * im
        
        # 1. Ancilla terms (Ancilla is bit N)
        ancilla_bit = (state >> N) & 1
        # Branchless calculation for M1 pipeline efficiency
        sz_A = 0.5 - ancilla_bit 
        diag_term += gA * sz_A
        
        # Sx_A term (off-diagonal: look at the state with flipped ancilla bit)
        out_val += hA * 0.5 * v_in[(state ⊻ ancilla_mask) + 1]
        
        # 2. Chain terms
        for j in 0:(N-1)
            # Branchless bit decoding
            sz_j = 0.5 - ((state >> j) & 1)
            sz_jp1 = 0.5 - ((state >> ((j + 1) % N)) & 1)
            
            # Diagonal terms: J S^z S^z + g S^z
            diag_term += J * sz_j * sz_jp1 + g * sz_j
            
            # Off-diagonal: h S^x (look at the state with flipped bit j)
            bit_j = 1 << j
            out_val += h * 0.5 * v_in[(state ⊻ bit_j) + 1]
            
            # Off-diagonal: lambda S^x S_A^x (flip both bit j and ancilla bit N)
            # Note the 0.25 factor properly accounts for spin-1/2 variables (1/2 * 1/2)
            out_val += lambda * 0.25 * v_in[(state ⊻ bit_j ⊻ ancilla_mask) + 1]
        end
        
        out_val += diag_term * amp_diag
        v_out[state + 1] = out_val
    end
    return v_out
end

"""
Builds the |+y> state tensor product.
"""
function build_initial_state(num_spins)
    num_states = 1 << num_spins
    v0 = zeros(ComplexF64, num_states)
    norm_factor = 1.0 / sqrt(2^num_spins)
    
    Threads.@threads for state in 0:(num_states-1)
        v0[state + 1] = norm_factor * (1im)^count_ones(state)
    end
    return v0
end

function compute_observables!(observables, t, v_t, v0, v0_A, v0_half, N, rho_half_buffer)
    # 1. Overall overlap
    push!(observables[:times], t)
    push!(observables[:overlap_overall], abs2(dot(v0, v_t)))
    
    # 2. Ancilla RDM
    state_matrix_A = reshape(v_t, (1 << N, 2))
    rho_A = state_matrix_A' * state_matrix_A 
    
    # Ancilla overlap, purity, and entropy
    push!(observables[:overlap_A], real(dot(v0_A, rho_A * v0_A)))
    
    evals_A = max.(eigvals(Hermitian(rho_A)), 1e-16)
    push!(observables[:purity_A], sum(evals_A.^2))
    
    entropy_A = -sum(evals_A .* log.(evals_A))
    push!(observables[:entropy_A], entropy_A)
    
    # 3. Half-chain RDM
    half_N = N ÷ 2
    state_tensor = reshape(v_t, (1 << half_N, 1 << half_N, 2))
    env_dim = (1 << half_N) * 2
    psi_bipartite = reshape(state_tensor, (1 << half_N, env_dim))
    
    # In-place matrix multiplication to avoid heavy memory allocation on every step
    mul!(rho_half_buffer, psi_bipartite, psi_bipartite')
    
    # Half-chain overlap, purity, and entropy
    push!(observables[:overlap_half], real(dot(v0_half, rho_half_buffer * v0_half)))
    
    evals_half = max.(eigvals(Hermitian(rho_half_buffer)), 1e-16)
    push!(observables[:purity_half], sum(evals_half.^2))
    entropy_half = -sum(evals_half .* log.(evals_half))
    push!(observables[:entropy_half], entropy_half)
    
    # Mutual Information I(A:System) = S(A) + S(Sys) - S(Total). Since state is pure, I = 2*S(A)
    return 2.0 * entropy_half - entropy_A 
end

function run_simulation(; N=10, J=1.0, h=1.05, lambda=0.2, hA=1.05, gA=0.1, g=0.1, dt=0.2, tol=5e-3, window_steps=200)

    # Initial states
    v0 = build_initial_state(N + 1)
    v0_A = build_initial_state(1)
    v0_half = build_initial_state(N ÷ 2)
    
    v_current = copy(v0)
    
    # Preallocate buffer for Half-Chain RDM to prevent memory thrashing
    half_N = N ÷ 2
    rho_half_buffer = zeros(ComplexF64, 1 << half_N, 1 << half_N)
    
    # Dynamic observables arrays
    observables = Dict(
        :times => Float64[],
        :overlap_overall => Float64[],
        :overlap_A => Float64[],
        :overlap_half => Float64[],
        :purity_A => Float64[],
        :entropy_A => Float64[],
        :purity_half => Float64[],
        :entropy_half => Float64[]
    )
    
    # Array to track Mutual Information for convergence check
    mi_history = Float64[]
    
    # Wrap Hamiltonian for KrylovKit
    H_action(v) = apply_H!(similar(v), v, N, J, h, lambda, hA, gA, g)
  #  H_action(v) = apply_H_transverse!(similar(v), v, N, J, h, lambda, hA, gA, g)
    
    t = 0.0
    step = 0
    converged = false
    
    while !converged
        # Compute observables for current state
        current_mi = compute_observables!(observables, t, v_current, v0, v0_A, v0_half, N, rho_half_buffer)
        push!(mi_history, current_mi)
        
        # Convergence Check based on Moving Average of Mutual Information
        if step >= 2 * window_steps
            # Grab views for memory efficiency
            window_current = @view mi_history[end - window_steps + 1 : end]
            window_prev = @view mi_history[end - 2*window_steps + 1 : end - window_steps]
            
            mean_current = mean(window_current)
            mean_prev = mean(window_prev)
            
            if abs(mean_current - mean_prev) < tol
                converged = true
                break # Exit the while loop
            end
        end
        
        # Evolve to next time step
        v_current, info = exponentiate(H_action, -1im * dt, v_current; ishermitian=true, tol=1e-8)
        
        t += dt
        step += 1
        
    end
    
    filename = "../../data/Svn_data/adaptive_IsingRA_Svn_N=$(N)_J=$(J)_lambda=$(lambda)_h=$(h)_g=$(g).jld2"
    
    # Ensure directory exists before saving (optional safeguard)
   # mkpath(dirname(filename))
    save_object(filename, observables)
end

N = 16
lambda = 0.1
h = 1.05
g = 0.0

dt = 0.1 / lambda

run_simulation(N=N, J=1.0, lambda=lambda, h=h, g=g, hA=h, gA=g, dt=dt, tol=1e-8, window_steps=400)






