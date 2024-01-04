int main() {

    /*
        Turns on "graph mode". Any enqueued programs will be written to DRAM, but
        not executed. All commands constructed will be cached. This API assumes
        that our command queues are in a start state (can potentially implicitly call
        finish and then reset the pointers)
    */
    BeginTrace(dispatch_queue=cq0, dataloader_queue=cq1);

    CreateInputNode(cq1, input_buffer0); // Notice this API does not take actual data, just the buffer
    CreateInputNode(cq1, input_buffer1);

    CreateProgramNode(cq0, program0); // Caches the enqueue program command and writes the program to DRAM
    CreateProgramNode(cq0, program1);
    ...
    CreateProgramNode(cq0, program100);

    CreateOutput(cq1, output_buffer0);

    /*
        Implicitly calls finish, resets the dispatch command queue pointers and re-writes the enqueue program
        commands into the hugepage. Ensures that the inputs are double-buffered.
    */
    CommandQueueGraph g = EndTrace(dispatch_queue=cq0, dataloader_queue=cq1);

    /*
        Events handled behind the scenes when running in graph mode.
    */
    for (int i = 0; i < 100; i++) {
        g.EnqueueWriteBuffer(0, data); // Writes data to buffer0 in a double buffered fashion.
        g.Enqueue();
    }

    EnqueueWriteBuffer(...); // Will assert, since in graph mode we can only use graph APIs
}
