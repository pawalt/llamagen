from prompting import get_llama_prompt

def get_refactor_prompt() -> str:
    instruction = """
I want to refactor error types for a function from error to cerr.Cerr. Refactor
all GRPC handlers to return cerr.Cerr instead of error. Do not return any text
other than the refactored text.

Perform this refactor by changing type signatures as follows:

Before:
func (s *Server) X(ctx context.Context, req *Y) (*Z, error)
After:
func (s *Server) X(ctx context.Context, req *Y) (*Z, cerr.Cerr)

Also modify errors so they're returning using cerr wrappers.
You have access to the following constructors:

package cerr
// New returns an error using the given explanation. It should only be used
// if you do not have a source error to wrap.
New(ctx context.Context, explanation string) Cerr
// Wrap wraps an error returned by another function using the given explanation.
// It should be used if you have an error returned by another function to wrap.
Wrap(ctx context.Context, explanation string, err error) Cerr

Examples:

Validation errors:

Before:
if len(users) == 0 {
    return nil, errors.New("cannot have no users")
}
After:
if len(users) == 0 {
    return nil, cerr.New(ctx, "cannot have no users")
}

Errors returned by other library:

Before:
_, err := someOp()
if err != nil {
    return nil, err
}
After:
_, err := someOp()
if err != nil {
    return nil, cerr.Wrap(ctx, "op failed", err)
}

Do not output anything except the refactored code.""".strip()
    
    code_to_refactor = """
func (s *Server) AddMethods(ctx context.Context, req *AddRequest) (*AddResponse, error) {
  res, err := add(req.X, req.Y)
  if err != nil {
    return nil, err
  }
  return &AddResponse{res}, nil
}

func add (x, y int) (int, error) {
  return x + y, nil
}
""".strip()
    
    return get_llama_prompt(instruction, code_to_refactor)