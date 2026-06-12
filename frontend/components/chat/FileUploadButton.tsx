"use client";
import { useDropzone } from "react-dropzone";
import { Spinner } from "@/components/ui/Spinner";

interface FileUploadButtonProps {
  onFile: (file: File) => void;
  uploading: boolean;
}

export function FileUploadButton({ onFile, uploading }: FileUploadButtonProps) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "application/pdf": [".pdf"] },
    maxSize: 10 * 1024 * 1024,
    multiple: false,
    onDropAccepted: (files) => onFile(files[0]),
  });

  return (
    <button
      {...getRootProps()}
      type="button"
      disabled={uploading}
      title="Upload PDF document"
      className="flex items-center justify-center w-8 h-8 rounded transition-colors cursor-pointer disabled:opacity-40"
      style={{
        color: isDragActive ? "#E8A020" : "#6B7E96",
        background: isDragActive ? "#3D2E0A" : "transparent",
      }}
      onMouseEnter={(e) => { if (!uploading) (e.currentTarget as HTMLButtonElement).style.color = "#F0F4F8"; }}
      onMouseLeave={(e) => { if (!uploading) (e.currentTarget as HTMLButtonElement).style.color = "#6B7E96"; }}
    >
      <input {...getInputProps()} />
      {uploading ? (
        <Spinner size={16} />
      ) : (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
        </svg>
      )}
    </button>
  );
}
