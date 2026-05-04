import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# 0. 기본 설정
st.set_page_config(layout="wide", page_title="Weld Line AI 통합 진단 시스템")
TARGET_VAR = 'Y_Weld'
DEFECT_THRESHOLD = 0.5

# 세션 초기화
if 'model' not in st.session_state:
    st.session_state.update({
        'model': None, 'scaler': None, 'df_weld': pd.DataFrame(),
        'ui_display_vars': [], 'global_process_vars': [],
        'global_bounds': {}, 'expert_constraints': {},
        'current_inputs': {}, 
        'ver': 0,
        'expert_reliability': 0.7, # 기본 신뢰성을 70%로 조정
        'last_res_val': None,
        'last_opt_df': None
    })

# =================================================================
# 1. 사이드바 및 데이터 로드
# =================================================================
with st.sidebar:
    st.header("📂 데이터 로드")
    u1 = st.file_uploader("1. UI 초기 조건 (기준값)", type=['csv','xlsx'])
    u2 = st.file_uploader("2. 가상 데이터", type=['csv','xlsx'])
    u3 = st.file_uploader("3. 해석 데이터", type=['csv','xlsx'])

    if st.button("🚀 AI 모델 학습 및 초기화", use_container_width=True):
        def load_data(f):
            if not f: return None
            return pd.read_csv(f) if f.name.endswith('csv') else pd.read_excel(f)
        
        df_i, df_v, df_r = load_data(u1), load_data(u2), load_data(u3)
        if df_i is not None and (df_v is not None or df_r is not None):
            # 데이터 병합 및 학습
            df_comb = pd.concat([df for df in [df_v, df_r] if df is not None]).dropna(subset=[TARGET_VAR])
            vars_list = [c for c in df_comb.columns if c != TARGET_VAR]
            df_comb[TARGET_VAR] = np.where(df_comb[TARGET_VAR] >= DEFECT_THRESHOLD, 1, 0)
            
            scaler = MinMaxScaler().fit(df_comb[vars_list])
            model = LogisticRegression().fit(scaler.transform(df_comb[vars_list]), df_comb[TARGET_VAR])
            
            # UI 표시용 변수 (초기 조건 파일 기준)
            ui_vars = [c for c in df_i.columns if c != TARGET_VAR]
            
            st.session_state.update({
                'model': model, 'scaler': scaler, 'df_weld': df_comb, 
                'global_process_vars': vars_list, 'ui_display_vars': ui_vars
            })
            
            # 💡 수정 포인트: 초기값을 기준으로 0% ~ 200% 범위 설정
            init_row = df_i.iloc[0].to_dict()
            for v in vars_list:
                base_val = float(init_row.get(v, 0))
                # 0~200% 범위 설정 (최소 0, 최대는 기준값의 2배)
                # 만약 기준값이 0이라면 범위를 0~100으로 임시 설정
                v_min = 0
                v_max = int(base_val * 2) if base_val > 0 else 100
                
                if v_min == v_max: v_max = v_min + 1
                
                st.session_state['global_bounds'][v] = (v_min, v_max)
                st.session_state['current_inputs'][v] = int(base_val)
                
            st.rerun()

# =================================================================
# 2. 메인 UI 및 진단 로직
# =================================================================
st.title("🛡️ Weld Line AI 통합 진단 시스템")

if st.session_state['model']:
    t1, t2 = st.tabs(["📊 진단 및 최적화", "📋 데이터 확인"])

    with t1:
        # A. 공정 조건 입력
        st.header("A. 현재 공정 조건 입력 (범위: 기존값의 0% ~ 200%)")
        cols = st.columns(3)
        for i, var in enumerate(st.session_state['ui_display_vars']):
            with cols[i % 3]:
                # 저장된 0~200% 범위를 불러옴
                b_min, b_max = st.session_state['global_bounds'].get(var, (0, 100))
                st.session_state['current_inputs'][var] = st.slider(
                    f"{var}", 
                    min_value=int(b_min), 
                    max_value=int(b_max), 
                    value=int(st.session_state['current_inputs'].get(var, b_min)), 
                    step=1, 
                    key=f"sl_{var}_{st.session_state['ver']}"
                )

        st.markdown("---")
        
        # B. 전문가 노하우 및 신뢰성
        st.header("B. 전문가 노하우 및 신뢰성")
        selected_expert_vars = st.multiselect(
            "전문가 관리 변수 선택", 
            options=st.session_state['ui_display_vars'],
            default=list(st.session_state['expert_constraints'].keys())
        )
        
        if selected_expert_vars:
            cols_b = st.columns(len(selected_expert_vars))
            for i, v_name in enumerate(selected_expert_vars):
                with cols_b[i]:
                    st.subheader(f"[{v_name}] 기준값")
                    st.session_state['expert_constraints'].setdefault(v_name, {'limit': st.session_state['current_inputs'].get(v_name, 0)})
                    st.session_state['expert_constraints'][v_name]['limit'] = st.number_input(
                        f"목표치 설정 ({v_name})", 
                        value=int(st.session_state['expert_constraints'][v_name]['limit']), 
                        step=1, key=f"num_{v_name}"
                    )
        
        st.session_state['expert_reliability'] = st.slider("👨‍🏫 전문가 의견 반영 강도 (%)", 0, 100, int(st.session_state['expert_reliability']*100)) / 100.0
        
        if st.button("💾 전문가 설정 반영", use_container_width=True):
            st.session_state['last_res_val'] = None
            st.session_state['last_opt_df'] = None
            st.rerun()

        st.markdown("---")
        
        # C. 최종 진단 및 최적화
        st.header("C. 최종 진단 결과")
        c_btn1, c_btn2 = st.columns(2)
        
        def calculate_risk(input_vals_list):
            all_v = st.session_state['global_process_vars']
            df_input = pd.DataFrame([input_vals_list], columns=all_v)
            ai_prob = st.session_state['model'].predict_proba(st.session_state['scaler'].transform(df_input))[0, 1]
            
            penalty = 0
            for v, c in st.session_state['expert_constraints'].items():
                v_idx = list(all_v).index(v)
                diff_ratio = abs(input_vals_list[v_idx] - c['limit']) / (c['limit'] + 1e-9)
                penalty += diff_ratio
            
            rel = st.session_state['expert_reliability']
            final_score = ai_prob + (penalty * rel)
            return min(1.0, final_score)

        # [현재 진단]
        if c_btn1.button("🔍 현재 조건 진단하기", type="primary", use_container_width=True):
            all_v = st.session_state['global_process_vars']
            input_vals = [float(st.session_state['current_inputs'].get(v, 0)) for v in all_v]
            st.session_state['last_res_val'] = calculate_risk(input_vals)
            st.session_state['last_opt_df'] = None
            st.rerun()

        # [최적 공정 도출]
        if c_btn2.button("✨ 최적 공정 도출", use_container_width=True):
            all_v = st.session_state['global_process_vars']
            x0 = [float(st.session_state['current_inputs'].get(v, 0.0)) for v in all_v]
            bnds = [st.session_state['global_bounds'].get(v, (0, 100)) for v in all_v]
            
            res = minimize(calculate_risk, x0, method='L-BFGS-B', bounds=bnds)
            
            if res.success:
                opt_dict = {v: int(round(val)) for v, val in zip(all_v, res.x)}
                st.session_state['last_res_val'] = res.fun
                st.session_state['last_opt_df'] = pd.DataFrame([{v: opt_dict.get(v) for v in st.session_state['ui_display_vars']}])
                st.rerun()

        # 📊 결과 출력
        if st.session_state['last_res_val'] is not None:
            st.markdown("---")
            val = st.session_state['last_res_val']
            st.subheader(f"📊 최종 진단 위험도: {val * 100:.2f}%")
            st.progress(min(1.0, float(val)))
            
            if st.session_state['last_opt_df'] is not None:
                st.success("✨ AI 추천 최적 공정 조건 (기준값 대비 0~200% 범위 내 최적화)")
                st.table(st.session_state['last_opt_df'])
                
                if st.button("📥 최적 조건을 슬라이더에 즉시 적용"):
                    opt_row = st.session_state['last_opt_df'].iloc[0].to_dict()
                    for v, val in opt_row.items():
                        st.session_state['current_inputs'][v] = int(val)
                    st.session_state['ver'] += 1
                    st.session_state['last_res_val'] = None
                    st.session_state['last_opt_df'] = None
                    st.rerun()

    with t2:
        if not st.session_state['df_weld'].empty:
            st.dataframe(st.session_state['df_weld'].head(50))
else:
    st.info("💡 사이드바에서 파일을 업로드하고 'AI 모델 학습'을 먼저 눌러주세요.")
