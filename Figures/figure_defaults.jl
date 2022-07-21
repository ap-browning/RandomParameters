gr()
default()
default(
    fontfamily="Arial",
    tick_direction=:out,
    guidefontsize=9,
    annotationfontfamily="Arial",
    annotationfontsize=10,
    annotationhalign=:left,
    box=:on,
    msw=0.0,
    lw=1.5
)

alphabet = "abcdefghijklmnopqrstuvwxyz"

function add_plot_labels!(plt;offset=0)
    n = length(plt.subplots)
    for i = 1:n
        plot!(plt,subplot=i,title="($(alphabet[i+offset]))")
    end
    plot!(
        titlelocation = :left,
        titlefontsize = 10,
        titlefontfamily = "Arial"
    )
end

## Colours
col_std = RGB(36/255,36/255,36/255)
col_alt = RGB(255/255,149/255,0/255)
col_norm = RGB(88/255,86/255,214/255)
col_skew = RGB(255/255,59/255,48/255)

col_blue = RGB(88/255,86/255,214/255)
col_orng = RGB(255/255,59/255,48/255)
col_teal = RGB(48/255,176/255,199/255)
col_green = RGB(32/255,138/255,61/255)