from simrep.pydocument import *
from math import *


def postprocess(datadir, simdir, repdir):
  """ This is the place to call postprocessing scripts which produce plots or other data. 
  datadir is the directory where the data is.
  simdir is a python object providing easy acces to various Cactus data 
  stored in datadir.
  repdir is the place where the figures should be saved in by the scripts.

  For this example we simply copy a picture to repdir."""

  import sys, shutil, os
  src = os.path.join(os.path.dirname(__file__), 
                        '..', 'data', 'Butterfly_tongue.jpg')
  dst = os.path.join(repdir, 'butterfly.jpg')
  shutil.copyfile(src, dst)
#

def create_report(datadir, simdir, repdir):
  """ Here we define the content shown from this module in the report """


  content = ["""Content is provided in a python based mini-language defined in the module pydocument.
different elements are collected in a python list. Elements can be lists themselves. Complicated 
elements are created by calling predefined functions. For example, a bullet list or enumeration 
is created from a list of items by the ulist or olist function. Basic elements are
""",
  ulist([["Integer numbers, e.g.", 13],
         ["Float numbers like", 12.5, "or", floatnum(pi, digits=13)],
         [emph("Emphasized"), "text and", warn("warnings")]
       ]),
  """One can create tables from a list of lists using the table function. When a caption is provided,
the table is taken out of the normal text flow similar as in LaTex, else it is inserted directly in 
the text.""",
  table([["A", "B", "C"],
         [1.0, 2.1, 3.2],
         [warn("NAN"), 45, "missing value"]
        ], cap="Example of a table."),
  """Figures are added by the figure function. The figures should be stored in the document directory repdir
or a subfolder of it. The path is then given relative to the document directory. The extension should be omitted,
a suitable format will be chosen automatically, converting formats if necessarry.""",
  figure('butterfly', cap="Example figure of a butterfly tongue.")
]
  # We now create a subsection for the above
  # This will show up in the secondary navigation bar as "Example1"
  sub = subsection("Example Subsection", "Example 1", cont=content)

  # Lets create another one
  sub2 = subsection("More examples", "Example 2", cont = [
    listing("""One can display logfiles or similar by passing them as a string
to the listing function. Line breaks will be preserved and the text is rendered 
in a monospaced font. In addition, one can provide a list of bad words which 
are highlighted""", alarming=['high'])
  ])

  # Now we create a section from a list of subsections
  # This will show up in the top level navigation bar as "Example"
  sec = section("Example Section", "Example", subs=[sub, sub2])

  # Each module always returns a section and nothing else. Those will be assembled and 
  # converted to HTML by the simreport script.
  return sec
#

