int main() {

    /*
        Turns on "graph mode". Any enqueued programs will be written to DRAM, but
        not executed. All commands constructed will be cached. This API assumes
        that our command queues are in a start state (can potentially implicitly call
        finish and then reset the pointers)
    */
    CommandQueueGraph g = BeginTrace(dispatch_queue=cq0, dataloader_queue=cq1);

    CreateInputNode(g, input_buffer0); // Notice this API does not take actual data, just the buffer
    CreateInputNode(g, input_buffer1);

    CreateProgramNode(g, program0); // Caches the enqueue program command and writes the program to DRAM
    CreateProgramNode(g, program1);
    ...
    CreateProgramNode(g, program100);

    CreateOutput(g, output_buffer0);
    /*
        Implicitly calls finish, resets the dispatch command queue pointers and re-writes the enqueue program
        commands into the hugepage. Ensures that the inputs are double-buffered. Has to write the program twice
        given that the inputs are double-buffered.
    */
    EndTrace(g);

    /*
        Events handled behind the scenes when running in graph mode.
    */
    for (int i = 0; i < 100; i++) {
        EnqueueWriteInput(g, 0, data0); // Writes data to buffer0 in a double buffered fashion.
        EnqueueWriteInput(g, 1, data1); // Writes data to buffer1 in a double buffered fashion.
        EnqueueGraph(g); // Runs the graph
    }

    EnqueueWriteBuffer(...); // Will assert, since in graph mode we can only use graph APIs
}
