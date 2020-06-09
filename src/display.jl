struct EmacsDisplay <: AbstractDisplay end

function Base.display(d::EmacsDisplay, x)
    display(d, "image/png", x)
end

function register_EmacsDisplay_backend()
    for d in Base.Multimedia.displays
        if typeof(d) <: EmacsDisplay
            return
        end
    end
    # register as backend
    pushdisplay(EmacsDisplay())
    # popdisplay()
    nothing
end

register_EmacsDisplay_backend()
