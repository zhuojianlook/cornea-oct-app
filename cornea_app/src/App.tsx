import { ThemeProvider, createTheme, CssBaseline } from "@mui/material";
import { AppShell } from "./components/layout/AppShell";

// Dark theme mirrors the multipanelfigure app (primary accent #5a7fa8).
const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#5a7fa8" },
    secondary: { main: "#63a66a" },
    error: { main: "#ff453a" },
    background: { default: "#1c1c1e", paper: "#2c2c2e" },
    text: { primary: "#e5e5ea", secondary: "#8e8e93" },
  },
  typography: {
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    fontSize: 13,
  },
  components: {
    MuiButton: {
      defaultProps: { size: "small", disableElevation: true },
      styleOverrides: { root: { textTransform: "none", fontSize: "0.75rem" } },
    },
    MuiSelect: { defaultProps: { size: "small" } },
    MuiSlider: { defaultProps: { size: "small" } },
    MuiTextField: { defaultProps: { size: "small", variant: "outlined" } },
    MuiDialog: { styleOverrides: { paper: { backgroundImage: "none" } } },
  },
});

export default function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <AppShell />
    </ThemeProvider>
  );
}
