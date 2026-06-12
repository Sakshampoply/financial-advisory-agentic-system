import { ReactNode } from "react";

interface ButtonProps {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "danger";
  size?: "sm" | "md";
  disabled?: boolean;
  title?: string;
  type?: "button" | "submit";
  className?: string;
}

export function Button({
  children,
  onClick,
  variant = "ghost",
  size = "md",
  disabled,
  title,
  type = "button",
  className = "",
}: ButtonProps) {
  const base =
    "inline-flex items-center justify-center gap-1.5 rounded font-medium transition-colors cursor-pointer select-none disabled:opacity-40 disabled:cursor-not-allowed";

  const sizes = {
    sm: "px-2.5 py-1 text-xs",
    md: "px-3.5 py-1.5 text-sm",
  };

  const variants = {
    primary:
      "bg-[#E8A020] text-[#080C14] hover:bg-[#D49018] active:bg-[#C08010]",
    ghost:
      "text-[#6B7E96] hover:text-[#F0F4F8] hover:bg-[#0F1520] active:bg-[#1E2A3A]",
    danger:
      "text-[#EF4444] hover:bg-[#1E2A3A] active:bg-[#2E1010]",
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`${base} ${sizes[size]} ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  );
}
