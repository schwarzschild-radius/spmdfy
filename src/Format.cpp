#include <spmdfy/Format.hpp>

namespace spmdfy {
namespace format {

static clang::FileID createInMemoryFile(StringRef FileName,
                                        MemoryBuffer *Source,
                                        clang::SourceManager &Sources,
                                        clang::FileManager &Files,
                                        llvm::vfs::InMemoryFileSystem *MemFS) {
    MemFS->addFileNoOwn(FileName, 0, Source);
    return Sources.createFileID(Files.getFile(FileName),
                                clang::SourceLocation(), clang::SrcMgr::C_User);
}

static bool parseLineRange(StringRef Input, unsigned &FromLine,
                           unsigned &ToLine) {
    std::pair<StringRef, StringRef> LineRange = Input.split(':');
    return LineRange.first.getAsInteger(0, FromLine) ||
           LineRange.second.getAsInteger(0, ToLine);
}

static bool fillRanges(MemoryBuffer *Code,
                       std::vector<clang::tooling::Range> &Ranges) {
    IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
        new llvm::vfs::InMemoryFileSystem);
    clang::FileManager Files(clang::FileSystemOptions(), InMemoryFileSystem);
    clang::DiagnosticsEngine Diagnostics(
        IntrusiveRefCntPtr<clang::DiagnosticIDs>(new clang::DiagnosticIDs),
        new clang::DiagnosticOptions);
    clang::SourceManager Sources(Diagnostics, Files);
    clang::FileID ID = createInMemoryFile("<irrelevant>", Code, Sources, Files,
                                          InMemoryFileSystem.get());
    clang::SourceLocation Start =
        Sources.getLocForStartOfFile(ID).getLocWithOffset(0);
    clang::SourceLocation End;
    End = Sources.getLocForEndOfFile(ID);
    unsigned Offset = Sources.getFileOffset(Start);
    unsigned Length = Sources.getFileOffset(End) - Offset;
    Ranges.push_back(clang::tooling::Range(Offset, Length));
    return false;
}

bool format(StringRef FileName) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> CodeOrErr =
        MemoryBuffer::getFileAsStream(FileName);
    if (std::error_code EC = CodeOrErr.getError()) {
        errs() << EC.message() << "\n";
        return true;
    }
    std::unique_ptr<llvm::MemoryBuffer> Code = std::move(CodeOrErr.get());
    if (Code->getBufferSize() == 0) {
        errs() << "File is empty?\n";
        return false; // Empty files are formatted correctly.
    }
    std::vector<clang::tooling::Range> Ranges;
    if (fillRanges(Code.get(), Ranges))
        return true;
    llvm::Expected<clang::format::FormatStyle> FormatStyle =
        clang::format::getStyle( //"{BasedOnStyle: llvm, IndentWidth: 4}",
            clang::format::DefaultFormatStyle, FileName,
            clang::format::DefaultFallbackStyle, Code->getBuffer());
    if (!FormatStyle) {
        llvm::errs() << llvm::toString(FormatStyle.takeError()) << "\n";
        return true;
    }
    unsigned CursorPosition = 0;
    Replacements Replaces = sortIncludes(*FormatStyle, Code->getBuffer(),
                                         Ranges, FileName, &CursorPosition);
    auto ChangedCode =
        clang::tooling::applyAllReplacements(Code->getBuffer(), Replaces);
    if (!ChangedCode) {
        llvm::errs() << llvm::toString(ChangedCode.takeError()) << "\n";
        return true;
    }
    // Get new affected ranges after sorting `#includes`.
    Ranges = clang::tooling::calculateRangesAfterReplacements(Replaces, Ranges);
    clang::format::FormattingAttemptStatus Status;
    Replacements FormatChanges =
        reformat(*FormatStyle, *ChangedCode, Ranges, FileName, &Status);
    Replaces = Replaces.merge(FormatChanges);
    IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
        new llvm::vfs::InMemoryFileSystem);
    clang::FileManager Files(clang::FileSystemOptions(), InMemoryFileSystem);
    clang::DiagnosticsEngine Diagnostics(
        IntrusiveRefCntPtr<clang::DiagnosticIDs>(new clang::DiagnosticIDs),
        new clang::DiagnosticOptions);
    clang::SourceManager Sources(Diagnostics, Files);
    clang::FileID ID = createInMemoryFile(FileName, Code.get(), Sources, Files,
                                          InMemoryFileSystem.get());
    clang::Rewriter Rewrite(Sources, clang::LangOptions());
    clang::tooling::applyAllReplacements(Replaces, Rewrite);
    if (Rewrite.overwriteChangedFiles())
        return true;
    return false;
}
} // namespace spmdfy
} // namespace spmdfy