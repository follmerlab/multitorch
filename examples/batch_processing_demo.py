#!/usr/bin/env python3
"""
Phase 2 Batch Processing Example and Benchmarks
================================================

Demonstrates the batch API and measures speedup for parameter sweeps.

Usage:
    python examples/batch_processing_demo.py
    
Expected output:
    - Performance comparison table (batch vs sequential)
    - Memory usage statistics
    - Example plots (if matplotlib available)
"""
import time
import torch
from multitorch.api.calc import (
    preload_fixture,
    calcXAS_cached,
    calcXAS_batch,
    batch_parameter_sweep,
)


def benchmark_batch_vs_sequential(cache, N, nbins=2000):
    """Compare batch vs sequential performance for N spectra."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {N} spectra (nbins={nbins})")
    print(f"{'='*60}")
    
    # Prepare parameter arrays
    slater_vals = torch.linspace(0.7, 0.9, N)
    soc_vals = torch.linspace(0.9, 1.1, N)
    
    # Warmup (JIT compilation, GPU cache)
    _ = calcXAS_batch(cache, slater_vals[:2], soc_vals[:2], nbins=nbins, device="cpu")
    
    # Batch timing
    print(f"\nBatch mode (N={N})...")
    start = time.perf_counter()
    y_batch = calcXAS_batch(
        cache, slater_vals, soc_vals, nbins=nbins, device="cpu"
    )
    batch_time = time.perf_counter() - start
    
    # Sequential timing (use subset if N large)
    N_seq = min(N, 100)
    print(f"Sequential mode ({N_seq} samples for timing)...")
    start = time.perf_counter()
    y_seq_list = []
    for i in range(N_seq):
        _, y = calcXAS_cached(
            cache, 
            slater=float(slater_vals[i]),
            soc=float(soc_vals[i]),
            nbins=nbins,
            device="cpu",
        )
        y_seq_list.append(y)
    seq_time_subset = time.perf_counter() - start
    
    # Extrapolate to full N
    seq_time_full = seq_time_subset * (N / N_seq)
    avg_time_seq = seq_time_subset / N_seq
    avg_time_batch = batch_time / N
    
    # Results
    speedup = seq_time_full / batch_time
    
    print(f"\n{'Metric':<30} {'Sequential':<15} {'Batch':<15} {'Speedup'}")
    print(f"{'-'*75}")
    print(f"{'Total time (s):':<30} {seq_time_full:>14.3f} {batch_time:>14.3f}  {speedup:>6.2f}×")
    print(f"{'Time per spectrum (ms):':<30} {avg_time_seq*1000:>14.1f} {avg_time_batch*1000:>14.1f}  {speedup:>6.2f}×")
    
    # Memory estimate
    mem_batch_mb = y_batch.element_size() * y_batch.numel() / 1e6
    print(f"\n{'Memory (spectra only):':<30} {mem_batch_mb:>14.1f} MB")
    
    # Verify parity for a few samples
    print(f"\nParity check (first {min(N_seq, 5)} spectra):")
    for i in range(min(N_seq, 5)):
        max_diff = (y_batch[i] - y_seq_list[i]).abs().max()
        print(f"  Sample {i}: max difference = {max_diff:.2e}")
    
    return {
        "N": N,
        "batch_time": batch_time,
        "seq_time": seq_time_full,
        "speedup": speedup,
        "avg_time_seq": avg_time_seq,
        "avg_time_batch": avg_time_batch,
    }


def example_grid_search(cache):
    """Demonstrate 2D parameter grid search."""
    print(f"\n{'='*60}")
    print("Example: 2D Grid Search")
    print(f"{'='*60}")
    
    slater_range = torch.linspace(0.6, 1.0, 20)
    soc_range = torch.linspace(0.8, 1.2, 20)
    
    print(f"Grid: {len(slater_range)} × {len(soc_range)} = {len(slater_range)*len(soc_range)} spectra")
    
    start = time.perf_counter()
    y_grid, s_axis, c_axis = batch_parameter_sweep(
        cache,
        slater_range=slater_range,
        soc_range=soc_range,
        nbins=1000,
    )
    elapsed = time.perf_counter() - start
    
    print(f"Shape: {y_grid.shape}")
    print(f"Time: {elapsed:.2f} s ({elapsed*1000/y_grid.shape[0]/y_grid.shape[1]:.1f} ms/spectrum)")
    
    return y_grid, s_axis, c_axis


def example_monte_carlo(cache):
    """Demonstrate Monte Carlo uncertainty quantification."""
    print(f"\n{'='*60}")
    print("Example: Monte Carlo Uncertainty Quantification")
    print(f"{'='*60}")
    
    N = 1000
    slater_mean, slater_std = 0.85, 0.05
    soc_mean, soc_std = 1.0, 0.1
    
    slater_samples = torch.normal(slater_mean, slater_std, size=(N,))
    soc_samples = torch.normal(soc_mean, soc_std, size=(N,))
    
    print(f"Sampling {N} parameter combinations from Gaussian distributions")
    print(f"  Slater: μ={slater_mean}, σ={slater_std}")
    print(f"  SOC:    μ={soc_mean}, σ={soc_std}")
    
    start = time.perf_counter()
    y_batch = batch_parameter_sweep(
        cache,
        slater_grid=slater_samples,
        soc_grid=soc_samples,
        nbins=1000,
    )
    elapsed = time.perf_counter() - start
    
    mean_spectrum = y_batch.mean(dim=0)
    std_spectrum = y_batch.std(dim=0)
    
    print(f"\nShape: {y_batch.shape}")
    print(f"Time: {elapsed:.2f} s ({elapsed*1000/N:.1f} ms/spectrum)")
    print(f"Mean spectrum intensity: {mean_spectrum.max():.3f}")
    print(f"Std spectrum max: {std_spectrum.max():.3f}")
    
    return y_batch, mean_spectrum, std_spectrum


def example_autograd_optimization(cache):
    """Demonstrate gradient-based parameter optimization."""
    print(f"\n{'='*60}")
    print("Example: Autograd Parameter Optimization")
    print(f"{'='*60}")
    
    # Simulate "experimental" data
    slater_true = 0.82
    soc_true = 1.05
    _, y_exp = calcXAS_cached(cache, slater=slater_true, soc=soc_true, nbins=500)
    
    # Initial guess
    slater_fit = torch.tensor(0.75, requires_grad=True)
    soc_fit = torch.tensor(0.95, requires_grad=True)
    
    print(f"True values: slater={slater_true}, soc={soc_true}")
    print(f"Initial guess: slater={slater_fit.item():.2f}, soc={soc_fit.item():.2f}")
    
    # Simple gradient descent
    learning_rate = 0.01
    n_iters = 10
    
    print(f"\nOptimizing with gradient descent (lr={learning_rate}, {n_iters} iterations)...")
    
    for i in range(n_iters):
        # Create small batch around current params for stability
        slater_batch = slater_fit.unsqueeze(0)
        soc_batch = soc_fit.unsqueeze(0)
        
        y_fit = calcXAS_batch(cache, slater_batch, soc_batch, nbins=500)[0]
        
        loss = ((y_fit - y_exp) ** 2).sum()
        loss.backward()
        
        # Update with gradients
        with torch.no_grad():
            slater_fit -= learning_rate * slater_fit.grad
            soc_fit -= learning_rate * soc_fit.grad
            
            # Zero gradients for next iteration
            slater_fit.grad.zero_()
            soc_fit.grad.zero_()
        
        if i % 2 == 0:
            print(f"  Iter {i:2d}: loss={loss.item():.2e}, "
                  f"slater={slater_fit.item():.3f}, soc={soc_fit.item():.3f}")
    
    print(f"\nFinal values: slater={slater_fit.item():.3f}, soc={soc_fit.item():.3f}")
    print(f"Errors: Δslater={abs(slater_fit.item()-slater_true):.3f}, "
          f"Δsoc={abs(soc_fit.item()-soc_true):.3f}")


def main():
    """Run all examples and benchmarks."""
    print("Phase 2 Batch Processing Demo")
    print("=" * 60)
    
    # Preload fixture once
    print("\nLoading Ni d8 D4h fixture...")
    cache = preload_fixture("Ni", "ii", "d4h")
    print("✓ Fixture loaded")
    
    # Run benchmarks
    results = []
    for N in [10, 100, 1000]:
        result = benchmark_batch_vs_sequential(cache, N, nbins=2000)
        results.append(result)
    
    # Summary table
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"{'N':<10} {'Batch (s)':<12} {'Sequential (s)':<15} {'Speedup':<10} {'ms/spectrum'}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['N']:<10} {r['batch_time']:<12.2f} {r['seq_time']:<15.2f} "
              f"{r['speedup']:<10.2f}× {r['avg_time_batch']*1000:<10.1f}")
    
    # Run examples
    grid_result = example_grid_search(cache)
    mc_result = example_monte_carlo(cache)
    example_autograd_optimization(cache)
    
    print(f"\n{'='*60}")
    print("All examples completed successfully!")
    print(f"{'='*60}")
    
    # Optional: Save results
    try:
        import pickle
        results_dict = {
            "benchmarks": results,
            "grid_search": grid_result,
            "monte_carlo": mc_result,
        }
        with open("/tmp/batch_demo_results.pkl", "wb") as f:
            pickle.dump(results_dict, f)
        print("\n✓ Results saved to /tmp/batch_demo_results.pkl")
    except Exception as e:
        print(f"\n⚠ Could not save results: {e}")


if __name__ == "__main__":
    main()
