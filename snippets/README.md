# Code snippet directory

These snippets are example scripts of how you could (!) use `DISKLAB`. They are
meant as templates to start your own model design. They demonstrate the
capabilities of `DISKLAB`. All of them are used in the tutorial, which can be
found in:

    ../tutorial/


## ADVICE 1:

If you modify these snippets (which is highly encouraged!) you are **strongly
advized** to *first* copy the snippet to another directory (e.g. `../models/`) and
only then adapt that copied snippet. You may need to also copy the
`snippet_header.py` (but see advice 2 below).

The reason why this is particularly important is because the snippets in this
directory, and the plots they produce, are directly imported into the tutorial
latex document, if you re-latex the tutorial. If you would modify the snippets
directly in this directory, and then latex the tutorial again, then your
modified snippets and plots end up in (your copy of) the tutorial, which may not
be what you want!!


## ADVICE 2:

The snippets all start with a `from snippet_header ...` statement and end with
`finalize()`. What are these? The snippet_header.py is merely a convenient list of
import statements, so that the snippets do not have to start all with the usual
"import numpy as np" and stuff. You are encouraged to have a look at this header
file to see what it is doing (at least up to the '----' line). But you can also
do these imports "by hand", if you prefer that.  For example in the snippet
`snippet_viscdiskevol_1.py` the first line reads:

    from snippet_header import DiskRadialModel, np, plt, MS, year, au, finalize

Instead of this you could also replace this with the following lines:

    import numpy as np
    import matplotlib.pyplot as plt
    from disklab import DiskRadialModel
    from disklab.natconst import MS, year, au

If you use the `finalize()` command (see below) you also need to add:

    from snippet_header import finalize

But also the `finalize()` command (at the end of each snippet) can be omitted
and replaced with `plt.show()`. The `finalize()` command is just a convenience
thing, mostly for us (the developers), since it helps us to run all snippets at
once and create all plots in one go. In many cases there is also a
`results=(array1,array2)` or something similar as keyword in the
`finalize()`. This is for the developers and allows automated testing of
DISKLAB, ensuring that continued development does not accidently break things
that used to work. You (the user) can replace the `finalize()` command with
`plt.show()`.

We think that the `snippet_header.py` is a convenient way to import all you
need. But it is up to you (the user) to decide how you want to use DISKLAB.
