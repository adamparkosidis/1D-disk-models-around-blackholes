# run `make rerun` to recalculate all scripts
# run `make rerun ARGS=` to recalculate all scripts but show figures until closed
# run `make rerun ARGS=--hash` to recalculate all scripts and create the hashes, but no plots
# run `make rerun ARGS=-t` to recalculate all scripts, testing the results against the hashes
# run `make rerun SNIPPETS=*deadzone*.py` to recalculate only a subset of snippets
# run `make` to rerun only the changed ones
# run `make clean` to delete all figures
ARGS=-n
SNIPPETS=$(wildcard snippet*.py)
PDFS=$(wildcard fig*.pdf)
PYTHON=python

.PHONY: all rerun clean

all: $(PDFS)

define thing
PDF := $(1)
SCRAP := $$(lastword $$(subst _, ,$$(PDF)))
SCRIPT := $$(patsubst fig_%_$$(SCRAP),%.py, $$(PDF))

$$(PDF): $$(SCRIPT)
	$(PYTHON) $$< $(ARGS)
endef

$(foreach PDF, $(PDFS), $(eval $(call thing, $(PDF))))

rerun:
	@for F in $(SNIPPETS) ; do \
	echo $(PYTHON) $$F $(ARGS); \
	$(PYTHON) $$F $(ARGS) || exit 1; \
	done

clean:
	-rm $(PDFS)
