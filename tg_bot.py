from joblib import load
from aiogram import Bot, Dispatcher, types, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder
import asyncio
import re
from tools import *
from config import API_TOKEN
from datetime import datetime

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()
dp.include_router(router)

subscriptions = {}

class Form(StatesGroup):
    url = State()

# Keyboard builder
async def get_refresh_keyboard(account_id):
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text='🔄 Refresh Match Info', callback_data=f'refresh_{account_id}'))
    return builder.as_markup()

def get_match_acc_id_keyboard():
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text='Check account (Top 250)', callback_data='check_acc'))
    return builder.as_markup()

# Commands
@router.message(Command('start'))
async def start_command(message: types.Message):
    await message.answer('A bot for getting real-time data for Deadlock matches (top 250 leaderboard only).', reply_markup=get_match_acc_id_keyboard())

@router.message(Command('check'))
async def check_acc_id(message: types.Message, state: FSMContext):
    user_id = message.chat.id
    await bot.send_message(user_id, 'Enter steam profile url: ')
    await state.set_state(Form.url)


@router.callback_query(F.data == 'check_acc')
async def check_acc_id(callback_query: types.CallbackQuery, state: FSMContext):
    user_id = callback_query.message.chat.id
    await bot.send_message(user_id, 'Enter steam profile url: ')
    await state.set_state(Form.url)

@router.message(Form.url)
async def process_profile_url(message: types.Message, state: FSMContext):
    url = message.text
    if not re.match(r'^https?://steamcommunity\.com/(id/\w+|profiles/\d+)/?$', url):
        await message.answer('Invalid Steam URL. Please try again.')
        return

    account_id = get_steamid3(url)
    if account_id:
        active_match = get_match_account_id(account_id)
        if active_match is not None:
            text = await generate_match_text(active_match)
            keyboard = await get_refresh_keyboard(account_id)
            sent_message = await message.answer(text, parse_mode='HTML', reply_markup=keyboard)
            subscriptions[(message.chat.id, account_id)] = sent_message.message_id
        else:
            await message.answer('No active match found for this account.')
    else:
        await message.answer('Failed to retrieve Steam ID. Please check the URL.')
    await state.clear()

@router.callback_query(F.data.startswith('refresh_'))
async def refresh_match_info(callback_query: types.CallbackQuery):
    account_id = callback_query.data.split('_')[1]
    active_match = get_match_account_id(account_id)
    if active_match is not None:
        text = await generate_match_text(active_match)
        keyboard = await get_refresh_keyboard(account_id)
        await bot.edit_message_text(
            text,
            chat_id=str(callback_query.message.chat.id),
            message_id=callback_query.message.message_id,
            parse_mode='HTML',
            reply_markup=keyboard
        )
    else:
        await bot.edit_message_text('No active match found.', 
            chat_id=str(callback_query.message.chat.id),
            message_id=callback_query.message.message_id
        )

async def generate_match_text(match_info):
    gbc = load('models/model.joblib')
    match_id, net_worth_team_0, net_worth_team_1, match_score, hero_ids_0, hero_ids_1 = get_match_info(match_info)
    heroes_0 = ', '.join(get_heroes(hero_ids_0))
    heroes_1 = ', '.join(get_heroes(hero_ids_1))
    softmax_arr = get_match_predict(match_info, gbc)
    ind = softmax_arr.argmax()
    team = 'The Amber Hand' if ind == 0 else 'The Sapphire Flame'
    prob = softmax_arr[0][ind]
    last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    text = f'''
<b>🏆 Match Information 🏆</b>
<b>Match ID:</b> {match_id}  
<b>Average rating:</b> {match_score}  
<b>Winning Team:</b> {team} ({prob * 100:.2f}% probability)  

<b>The Amber Hand</b>  
- <b>Net Worth:</b> {net_worth_team_0}  
- <b>Heroes:</b> {heroes_0}  

<b>The Sapphire Flame</b>  
- <b>Net Worth:</b> {net_worth_team_1}  
- <b>Heroes:</b> {heroes_1}  

<b>Last Updated:</b> {last_updated}
'''
    return text

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())