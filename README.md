The Policy Algorithm of solving mean field game and mean field type control problem with finite difference method.

![](mfg.png)

Check the examples in [colab](https://colab.research.google.com/drive/19FLjyv5alw3dq1QeK9zovuAMfIM3LKPE?usp=sharing).

There are some code could be optimized to get more efficiency.

Now only support periodic boundary condition and 1d/2d problem.

the main code about policy iteration is as:

```{julia}
# println("start Policy Iteration")
for iter in 1:maxit
    solve_FP!(M, Q)
    update_control!(Q_tilde, U, M, D, update_Q)
    solve_HJB!(U, M, Q_tilde)
    update_control!(Q_new, U, M, D, update_Q)

    resFP, resHJB = compute_res(U, M, Q_new)
    Q, Q_new = Q_new, Q

    ### other logging code
end
```

the helper function like `solve_FP!` and others could be found in `src/utils.jl`
