#include <clang/AST/Decl.h>

namespace spmdfy{

class ISPCGenerator{
    public:
        virtual ~ISPCGenerator() = default;
        // every generator must override this to handle translation unit...
        virtual auto handleTranslationUnit(clang::ASTContext&) -> bool = 0; // !OVERRIDE THIS
    private:
};

}