#ifndef SPMDFY_GENERATOR_HPP
#define SPMDFY_GENERATOR_HPP

#include <clang/AST/Decl.h>

namespace spmdfy {
/**
 * \class ISPCGenerator
 * \ingroup CodeGen
 *
 * \brief An Interface class to the generator which the frontend invokes by
 * passing the AST Context
 *
 * */
class ISPCGenerator {
  public:
    virtual ~ISPCGenerator() = default;
    // every generator must override this to handle translation unit...

    /// handles the the CUDA Translation Unit
    /// \param AST context of the Translation Unit
    virtual auto handleTranslationUnit(clang::ASTContext &)
        -> bool = 0; // !OVERRIDE THIS
  private:
};

} // namespace spmdfy

#endif