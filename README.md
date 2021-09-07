The Policy Algorithm of solving mean field game and mean field type control problem with finite difference method. 

$$
\left\{\begin{array}{c}
-\frac{\partial u}{\partial t}-\varepsilon \Delta u+\frac{1}{2F_1(m)}|\nabla u|^{2} -V(x) - F_2(m)= 0 \\
\frac{\partial m}{\partial t}-\varepsilon \Delta m- \operatorname{div}\left(\frac{m \nabla u}{F_1(m)}\right)=0
\end{array}\right.
$$

Check the examples in [colab](https://colab.research.google.com/drive/19FLjyv5alw3dq1QeK9zovuAMfIM3LKPE?usp=sharing).

There are some code could be optimized to get more efficiency.

Now only support periodic boundary contion and 1d/2d problem.

