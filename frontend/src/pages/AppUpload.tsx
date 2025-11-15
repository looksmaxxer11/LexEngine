import { OCRUpload } from "../components/OCRUpload";

export function AppUpload() {
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 break-words">Upload Document</h1>
        <p className="text-slate-600 mt-2">
          Upload a PDF document to extract text with our advanced OCR
        </p>
      </div>
      <OCRUpload />
    </div>
  );
}
