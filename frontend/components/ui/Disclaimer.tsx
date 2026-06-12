export function Disclaimer({ text }: { text: string }) {
  return (
    <p className="text-xs italic text-[#6B7E96] mt-3 pt-3 border-t border-[#1E2A3A]">
      {text}
    </p>
  );
}
