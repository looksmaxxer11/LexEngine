-- OCR SaaS Database Schema for Supabase
-- Run these SQL commands in your Supabase SQL Editor

-- Table: ocr_results
-- Stores OCR processing results
CREATE TABLE IF NOT EXISTS ocr_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    text TEXT NOT NULL,
    pages INTEGER NOT NULL DEFAULT 0,
    columns INTEGER NOT NULL DEFAULT 0,
    characters INTEGER NOT NULL DEFAULT 0,
    processing_time NUMERIC NOT NULL DEFAULT 0,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Table: user_stats
-- Tracks user processing statistics
CREATE TABLE IF NOT EXISTS user_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID UNIQUE NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    total_pages INTEGER NOT NULL DEFAULT 0,
    total_documents INTEGER NOT NULL DEFAULT 0,
    last_processed TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_ocr_results_user_id ON ocr_results(user_id);
CREATE INDEX IF NOT EXISTS idx_ocr_results_created_at ON ocr_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_stats_user_id ON user_stats(user_id);

-- Row Level Security (RLS) Policies
ALTER TABLE ocr_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_stats ENABLE ROW LEVEL SECURITY;

-- Policy: Users can view their own OCR results
CREATE POLICY "Users can view own results" ON ocr_results
    FOR SELECT
    USING (auth.uid() = user_id OR user_id IS NULL);

-- Policy: Users can insert their own OCR results
CREATE POLICY "Users can insert own results" ON ocr_results
    FOR INSERT
    WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

-- Policy: Users can delete their own OCR results
CREATE POLICY "Users can delete own results" ON ocr_results
    FOR DELETE
    USING (auth.uid() = user_id);

-- Policy: Users can view their own stats
CREATE POLICY "Users can view own stats" ON user_stats
    FOR SELECT
    USING (auth.uid() = user_id);

-- Policy: Users can insert their own stats
CREATE POLICY "Users can insert own stats" ON user_stats
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Policy: Users can update their own stats
CREATE POLICY "Users can update own stats" ON user_stats
    FOR UPDATE
    USING (auth.uid() = user_id);

-- Function: Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Auto-update updated_at for ocr_results
CREATE TRIGGER update_ocr_results_updated_at
    BEFORE UPDATE ON ocr_results
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger: Auto-update updated_at for user_stats
CREATE TRIGGER update_user_stats_updated_at
    BEFORE UPDATE ON user_stats
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Anonymous access policy (allow unauthenticated users to save results)
CREATE POLICY "Anonymous can insert results" ON ocr_results
    FOR INSERT
    WITH CHECK (user_id IS NULL);

CREATE POLICY "Anonymous can view own results" ON ocr_results
    FOR SELECT
    USING (user_id IS NULL);

-- Grant permissions
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ocr_results TO anon, authenticated;
GRANT ALL ON user_stats TO authenticated;
