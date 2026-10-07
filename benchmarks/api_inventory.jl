using FEMTools
using TOML

"""
    main(; output_path=nothing)

Inventory FEMTools-owned public names, runtime methods, and repository textual
references. References include definitions, docstrings, and comments; inspect
them before removing a method. This host-only development tool does not run
solvers or transfer numerical fields. An optional TOML report belongs outside
the repository.
"""
function main(; output_path = nothing)
    root = dirname(@__DIR__)
    files = [joinpath(root, "README.md")]
    for directory in ("src", "test", "docs/src", "examples", "benchmarks", "ext")
        for (path, _, names) in walkdir(joinpath(root, directory))
            append!(files, [joinpath(path, name) for name in names
                            if endswith(name, ".jl") || endswith(name, ".md")])
        end
    end
    sources = [(replace(relpath(file, root), '\\' => '/'), readlines(file))
               for file in sort(files)]
    inventory = Dict{String, Any}[]
    for name in sort(names(FEMTools; all = true))
        name === :FEMTools && continue
        Base.ispublic(FEMTools, name) || continue
        value = getfield(FEMTools, name)
        signatures = value isa Function || value isa Type ? string.(collect(methods(value))) : String[]
        # Public names are identifiers, optionally ending in !. Identifier
        # boundaries avoid counting pressure_mass as a reference to pressure.
        pattern = Regex("(?<![\\p{L}\\p{N}_])" * string(name) * "(?![\\p{L}\\p{N}_!])")
        references = ["$file:$line" for (file, lines) in sources
                      for (line, source) in enumerate(lines) if occursin(pattern, source)]
        push!(inventory, Dict("name" => string(name),
                              "tier" => Base.isexported(FEMTools, name) ? "exported" : "public",
                              "methods" => signatures, "references" => references))
    end
    report = Dict("julia" => string(VERSION),
                  "package_version" => string(pkgversion(FEMTools)),
                  "reference_kind" => "textual, not a static call graph",
                  "api" => inventory)
    if output_path !== nothing
        open(output_path, "w") do io
            TOML.print(io, report; sorted = true)
        end
    end
    println("Exported names: ", count(item -> item["tier"] == "exported", inventory))
    println("Public unexported names: ", count(item -> item["tier"] == "public", inventory))
    println("Methods: ", sum(item -> length(item["methods"]), inventory))
    return report
end

main(; output_path = isempty(ARGS) ? nothing : only(ARGS))
