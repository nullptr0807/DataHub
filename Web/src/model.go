package main

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
	Mode         string  `json:"mode"`
	Data         string  `json:"data"`
	Lower        float64 `json:"lower"`
	Upper        float64 `json:"upper"`
	Levels       int     `json:"levels"`
	MinLevels    int     `json:"min_levels"`
	MaxLevels    int     `json:"max_levels"`
	LevelStep    int     `json:"level_step"`
	Fee          float64 `json:"fee"`
	QuoteBalance float64 `json:"quote_balance"`
	BaseBalance  float64 `json:"base_balance"`
	DaysBack     int     `json:"days_back"`
	RiskFree     float64 `json:"risk_free"`
}

type tradePoint struct {
	Timestamp string  `json:"timestamp"`
	Type      string  `json:"type"`
	Price     float64 `json:"price"`
	Quantity  float64 `json:"quantity"`
	Level     float64 `json:"level"`
}

type candlePoint struct {
	Timestamp string  `json:"timestamp"`
	Open      float64 `json:"open"`
	High      float64 `json:"high"`
	Low       float64 `json:"low"`
	Close     float64 `json:"close"`
}

type equityPoint struct {
	Timestamp string  `json:"timestamp"`
	Equity    float64 `json:"equity"`
}

type backtestResult struct {
	Equity        float64       `json:"equity"`
	InitialEquity float64       `json:"initial_equity"`
	ShareValue    float64       `json:"share_value"`
	QuoteBalance  float64       `json:"quote_balance"`
	BaseBalance   float64       `json:"base_balance"`
	BuyCount      int           `json:"buy_count"`
	SellCount     int           `json:"sell_count"`
	SkippedBuys   int           `json:"skipped_buys"`
	SkippedSells  int           `json:"skipped_sells"`
	InitialPrice  float64       `json:"initial_price"`
	FinalPrice    float64       `json:"final_price"`
	ROI           float64       `json:"roi"`
	PriceReturn   float64       `json:"price_return"`
	Alpha         float64       `json:"alpha"`
	SharpeRatio   *float64      `json:"sharpe_ratio"`
	RiskFreeRate  float64       `json:"risk_free_rate"`
	GridLevels    []float64     `json:"grid_levels"`
	Candles       []candlePoint `json:"candles"`
	Trades        []tradePoint  `json:"trades"`
	EquityCurve   []equityPoint `json:"equity_curve"`
}

type rangeResult struct {
	Levels        int      `json:"levels"`
	ROI           float64  `json:"roi"`
	PriceReturn   float64  `json:"price_return"`
	Alpha         float64  `json:"alpha"`
	SharpeRatio   *float64 `json:"sharpe_ratio"`
	InitialEquity float64  `json:"initial_equity"`
	FinalEquity   float64  `json:"final_equity"`
	QuoteBalance  float64  `json:"quote_balance"`
	BaseBalance   float64  `json:"base_balance"`
	BuyCount      int      `json:"buy_count"`
	SellCount     int      `json:"sell_count"`
	SkippedBuys   int      `json:"skipped_buys"`
	SkippedSells  int      `json:"skipped_sells"`
}

type rangeSummary struct {
	BestROI    rangeResult  `json:"best_roi"`
	BestSharpe *rangeResult `json:"best_sharpe,omitempty"`
}

type apiResponse struct {
	Success      bool            `json:"success"`
	Error        string          `json:"error,omitempty"`
	Mode         string          `json:"mode,omitempty"`
	UnitResult   *backtestResult `json:"unit_result,omitempty"`
	RangeResults []rangeResult   `json:"range_results,omitempty"`
	RangeSummary *rangeSummary   `json:"range_summary,omitempty"`
	RawOutput    string          `json:"raw_output,omitempty"`
}
