#=
    plotting.jl
    Contains code to produce model related plots.
    Author:     Alexander P. Browning
                ======================
                School of Mathematical Sciences
                Queensland University of Technology
                ======================
                ap.browning@icloud.com
                alexbrowning.me
=# 
"""
    density2d(x,y,...)
Create 2D kernel density plot.
"""
@userplot density2d
@recipe function f(kc::density2d; trim=0, levels=10, clip=((-3.0, 3.0), (-3.0, 3.0)))
    x,y = kc.args

    x = vec(x)
    y = vec(y)

    k = KernelDensity.kde((x, y)) 

    legend --> false

    @series begin
        seriestype := contourf
        colorbar := false
        (collect(k.x), collect(k.y), k.density')
    end

end