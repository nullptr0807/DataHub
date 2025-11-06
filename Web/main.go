package main

import (
    "encoding/json"
    "errors"
    "fmt"
    "html/template"
    "log"
    "net/http"
    "os"
    "os/exec"
    "path/filepath"
    "sort"
    "strconv"
    "strings"
)

type pageData struct {
    CSVFiles        []csvOption
    DefaultData     string
    DefaultRiskFree float64
    DefaultLevels   int
    DefaultDaysBack int
}

type csvOption struct {
    Value string
    Label string
}

type backtestRequest struct {
    Data         string  `json:"data"`
    Lower        float64 `json:"lower"`
    Upper        float64 `json:"upper"`
    Levels       int     `json:"levels"`
    Fee          float64 `json:"fee"`
    QuoteBalance float64 `json:"quote_balance"`
    BaseBalance  float64 `json:"base_balance"`
    DaysBack     int     `json:"days_back"`
    RiskFree     float64 `json:"risk_free"`
}

type backtestResult struct {
    Equity        float64  `json:"equity"`
    InitialEquity float64  `json:"initial_equity"`
    ShareValue    float64  `json:"share_value"`
    QuoteBalance  float64  `json:"quote_balance"`
    BaseBalance   float64  `json:"base_balance"`
    BuyCount      int      `json:"buy_count"`
    SellCount     int      `json:"sell_count"`
    SkippedBuys   int      `json:"skipped_buys"`
    SkippedSells  int      `json:"skipped_sells"`
    InitialPrice  float64  `json:"initial_price"`
    FinalPrice    float64  `json:"final_price"`
    ROI           float64  `json:"roi"`
    PriceReturn   float64  `json:"price_return"`
    Alpha         float64  `json:"alpha"`
    SharpeRatio   *float64 `json:"sharpe_ratio"`
    RiskFreeRate  float64  `json:"risk_free_rate"`
}

type apiResponse struct {
    Success   bool            `json:"success"`
    Error     string          `json:"error,omitempty"`
    Result    *backtestResult `json:"result,omitempty"`
    RawOutput string          `json:"raw_output,omitempty"`
}

var (
    repoRoot      string
    pythonBinary  string
    tmplIndex     *template.Template
    defaultData   = filepath.ToSlash(filepath.Join("Collector", "data", "btc_1m.csv"))
    defaultLevels = 21
    defaultDays   = 90
    defaultRisk   = 0.02
)

func main() {
    if err := run(); err != nil {
        log.Fatal(err)
    }
}

func run() error {
    wd, err := os.Getwd()
    if err != nil {
        return err
    }

    repoRootCandidate := filepath.Dir(wd)
    if filepath.Base(wd) != "web" {
        repoRootCandidate = wd
    }

    repoRootAbs, err := filepath.Abs(repoRootCandidate)
    if err != nil {
        return err
    }
    repoRoot = repoRootAbs

    pythonBinary = filepath.Join(repoRoot, ".venv", "bin", "python")
    if _, err := os.Stat(pythonBinary); err != nil {
        return fmt.Errorf("python interpreter not found at %s: %w", pythonBinary, err)
    }
    if _, err := os.Stat(filepath.Join(repoRoot, "Backtest", "GridBasic.py")); err != nil {
        return fmt.Errorf("GridBasic.py not found: %w", err)
    }

    templatePath := filepath.Join(repoRoot, "web", "templates", "index.html")
    parsed, err := template.ParseFiles(templatePath)
    if err != nil {
        return fmt.Errorf("failed to parse template %s: %w", templatePath, err)
    }
    tmplIndex = parsed

    http.HandleFunc("/", handleIndex)
    http.HandleFunc("/api/backtest", handleBacktest)

    addr := ":8080"
    log.Printf("Serving DataHub UI at http://localhost%s", addr)
    return http.ListenAndServe(addr, nil)
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
    options, err := listCSVOptions()
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    defaultDataValue := defaultData
    if len(options) > 0 {
        found := false
        for _, opt := range options {
            if opt.Value == defaultData {
                defaultDataValue = opt.Value
                found = true
                break
            }
        }
        if !found {
            defaultDataValue = options[0].Value
        }
    }

    data := pageData{
        CSVFiles:        options,
        DefaultData:     defaultDataValue,
        DefaultRiskFree: defaultRisk,
        DefaultLevels:   defaultLevels,
        DefaultDaysBack: defaultDays,
    }

    if err := tmplIndex.Execute(w, data); err != nil {
        log.Printf("template error: %v", err)
    }
}

func handleBacktest(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
        return
    }

    decoder := json.NewDecoder(r.Body)
    decoder.DisallowUnknownFields()

    req := backtestRequest{}
    if err := decoder.Decode(&req); err != nil {
        respondJSON(w, http.StatusBadRequest, apiResponse{Success: false, Error: fmt.Sprintf("invalid payload: %v", err)})
        return
    }

    applyDefaults(&req)
    if err := validateRequest(req); err != nil {
        respondJSON(w, http.StatusBadRequest, apiResponse{Success: false, Error: err.Error()})
        return
    }

    result, raw, err := runBacktest(req)
    if err != nil {
        respondJSON(w, http.StatusInternalServerError, apiResponse{Success: false, Error: err.Error(), RawOutput: raw})
        return
    }

    respondJSON(w, http.StatusOK, apiResponse{Success: true, Result: &result})
}

func applyDefaults(req *backtestRequest) {
    if req.Levels <= 0 {
        req.Levels = defaultLevels
    }
    if req.DaysBack <= 0 {
        req.DaysBack = defaultDays
    }
    if req.Fee < 0 {
        req.Fee = 0
    }
    if req.QuoteBalance <= 0 {
        req.QuoteBalance = 10000
    }
    if req.RiskFree <= 0 {
        req.RiskFree = defaultRisk
    }
    if req.Data == "" {
        req.Data = defaultData
    }
}

func validateRequest(req backtestRequest) error {
    if req.Lower <= 0 || req.Upper <= 0 {
        return errors.New("bounds must be positive")
    }
    if req.Lower >= req.Upper {
        return errors.New("lower bound must be below upper bound")
    }
    if req.Levels < 2 {
        return errors.New("levels must be at least 2")
    }
    if req.DaysBack < 0 {
        return errors.New("days_back cannot be negative")
    }
    return nil
}

func runBacktest(req backtestRequest) (backtestResult, string, error) {
    dataPath, err := resolveDatasetPath(req.Data)
    if err != nil {
        return backtestResult{}, "", err
    }

    args := []string{
        filepath.Join("Backtest", "GridBasic.py"),
        "--data", dataPath,
        "--lower", fmt.Sprintf("%f", req.Lower),
        "--upper", fmt.Sprintf("%f", req.Upper),
        "--levels", strconv.Itoa(req.Levels),
        "--fee", fmt.Sprintf("%f", req.Fee),
        "--quote-balance", fmt.Sprintf("%f", req.QuoteBalance),
        "--base-balance", fmt.Sprintf("%f", req.BaseBalance),
        "--risk-free", fmt.Sprintf("%f", req.RiskFree),
        "--output-format", "json",
    }
    if req.DaysBack > 0 {
        args = append(args, "--days-back", strconv.Itoa(req.DaysBack))
    }

    cmd := exec.Command(pythonBinary, args...)
    cmd.Dir = repoRoot
    cmd.Env = append(os.Environ(), fmt.Sprintf("PYTHONPATH=%s", repoRoot))

    output, err := cmd.CombinedOutput()
    raw := strings.TrimSpace(string(output))
    if err != nil {
        return backtestResult{}, raw, fmt.Errorf("backtest failed: %w", err)
    }

    var result backtestResult
    if err := json.Unmarshal(output, &result); err != nil {
        return backtestResult{}, raw, fmt.Errorf("failed to parse backtest output: %w", err)
    }

    return result, raw, nil
}

func listCSVOptions() ([]csvOption, error) {
    dataDir := filepath.Join(repoRoot, "Collector", "data")
    entries, err := os.ReadDir(dataDir)
    if err != nil {
        if errors.Is(err, os.ErrNotExist) {
            return nil, nil
        }
        return nil, err
    }

    options := make([]csvOption, 0, len(entries))
    for _, entry := range entries {
        if entry.IsDir() {
            continue
        }
        if !strings.EqualFold(filepath.Ext(entry.Name()), ".csv") {
            continue
        }
        rel := filepath.ToSlash(filepath.Join("Collector", "data", entry.Name()))
        options = append(options, csvOption{Value: rel, Label: entry.Name()})
    }

    sort.Slice(options, func(i, j int) bool {
        return options[i].Label < options[j].Label
    })

    return options, nil
}

func resolveDatasetPath(rel string) (string, error) {
    clean := filepath.Clean(rel)
    if strings.HasPrefix(clean, "..") {
        return "", errors.New("dataset path must stay under Collector/data")
    }

    dataDir := filepath.Join(repoRoot, "Collector", "data")
    abs := filepath.Join(repoRoot, clean)
    if !strings.HasPrefix(abs, dataDir) {
        return "", errors.New("dataset must be inside Collector/data")
    }
    if _, err := os.Stat(abs); err != nil {
        return "", fmt.Errorf("dataset not found: %w", err)
    }

    return abs, nil
}

func respondJSON(w http.ResponseWriter, status int, payload apiResponse) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    if err := json.NewEncoder(w).Encode(payload); err != nil {
        log.Printf("encode error: %v", err)
    }
}
