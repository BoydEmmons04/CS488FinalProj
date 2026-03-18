import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modeling import predict_fares
from flask import Flask, render_template