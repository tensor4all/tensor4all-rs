"""Return a simple worst-case bond-dimension envelope.

`num_bonds` is the number of bond cuts in the plotted profile. The returned
vector has one entry per bond cut.
"""
function worst_case_bond_dims(num_bonds; base = 2)
    num_sites = num_bonds + 1
    half = num_sites ÷ 2
    up = [base^x for x in 1:half]
    down = reverse(up)

    if length(up) + length(down) >= num_sites
        down = down[2:end]
    end

    return [up..., down...]
end

"""Overlay a dashed worst-case envelope on a bond-dimension axis."""
function add_worst_case_envelope!(ax, bond_index; base = 2, label = "worst case")
    lines!(
        ax,
        bond_index,
        worst_case_bond_dims(length(bond_index); base = base),
        color = :gray40,
        linewidth = 2,
        linestyle = Linestyle([0, 10, 15]),
        label = label,
    )
    return ax
end
