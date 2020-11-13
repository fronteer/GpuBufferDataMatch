HIPCC=/opt/rocm/hip/bin/hipcc

CFLAGS := 

PROGRAMS :=  bufferMatch_test 

all: $(PROGRAMS)

# Step

bufferMatch_test:  bufferMatch.o
	$(HIPCC) -o $@ $< 

%.o: %.cpp
	$(HIPCC) $(CFLAGS) -c -o $@ $< 


clean:
	rm -f *.o $(PROGRAMS)

