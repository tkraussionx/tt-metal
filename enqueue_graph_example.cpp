int main() {

    /*
        Turns on "graph mode". Any enqueued programs will be written to DRAM, but
        not executed. All commands constructed will be cached. This API assumes
        that our command queues are in a start state (can potentially implicitly call
        finish and then reset the pointers)
    */
    CommandQueueGraph g = BeginTrace(dispatch_queue=cq0, dataloader_queue=cq1);

    CreateInputNode(input_buffer0); // Notice this API does not take actual data, just the buffer
    CreateInputNode(input_buffer1);

    CreateProgramNode(program0); // Caches the enqueue program command and writes the program to DRAM
    CreateProgramNode(program1);
    ...
    CreateProgramNode(program100);

    CreateOutput(output_buffer0);
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
        g.EnqueueWriteBuffer(0, data); // Writes data to buffer0 in a double buffered fashion.
        g.Enqueue(); // Runs the graph
    }

    EnqueueWriteBuffer(...); // Will assert, since in graph mode we can only use graph APIs
}
