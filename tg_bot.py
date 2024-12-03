from joblib import load
import pandas as pd
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
from config import DB_SETTINGS
import asyncpg
import atexit

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()
dp.include_router(router)


# Postgres -------------------

db_pool = None

async def create_db_pool():
    return await asyncpg.create_pool(**DB_SETTINGS)

async def save_subscription(user_id, account_id):
    async with db_pool.acquire() as conn:
        await conn.execute('''
            INSERT INTO subscriptions (user_id, account_id)
            VALUES ($1, $2)
            ON CONFLICT (user_id, account_id) DO NOTHING
        ''', user_id, account_id)

async def update_message_id(user_id, account_id, message_id):
    async with db_pool.acquire() as conn:
        await conn.execute('''
            UPDATE subscriptions
            SET message_id = $3
            WHERE user_id = $1 AND account_id = $2
        ''', user_id, account_id, message_id)

async def get_all_subscriptions():
    async with db_pool.acquire() as conn:
        rows = await conn.fetch('SELECT user_id, account_id, message_id FROM subscriptions')
        return [(row['user_id'], row['account_id'], row['message_id']) for row in rows]
    
async def delete_subscription(user_id, account_id):
    async with db_pool.acquire() as conn:
        await conn.execute('''
            DELETE FROM subscriptions
            WHERE user_id = $1 AND account_id = $2
        ''', user_id, account_id)
        
# Postgres -------------------


class Form(StatesGroup):
    url = State()

def get_match_acc_id_keyboard():
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text='Link account (Top 250)', callback_data='link_acc'))
    return builder.as_markup()

# Commands -------------------

@router.message(Command('start'))
async def start_command(message: types.Message):
    user_id = message.chat.id
    await message.answer('A bot for getting real-time data for Deadlock matches (top 250 leaderboard only).', reply_markup=get_match_acc_id_keyboard())

@router.message(Command('link'))
async def link_acc_id(message: types.Message, state: FSMContext):
    user_id = message.chat.id
    await bot.send_message(user_id, 'Enter steam profile url: ')
    await state.set_state(Form.url)

@router.message(Command('unsubscribe'))
async def unsubscribe_command(message: types.Message):
    user_id = message.chat.id
    await delete_subscription(user_id, None)
    await message.answer('Unsubscribed from all accounts.')

# Commands -------------------


@router.callback_query(F.data == 'link_acc')
async def link_acc_id(callback_query: types.CallbackQuery, state: FSMContext):
    user_id = callback_query.message.chat.id
    await bot.send_message(user_id, 'Enter steam profile url: ')
    await state.set_state(Form.url)

@router.message(Form.url)
async def subscribe_to_account(message: types.Message, state: FSMContext):
    url = message.text
    if not re.match(r'^https?://steamcommunity\.com/(id/\w+|profiles/\d+)/?$', url):
        await message.answer('Invalid Steam URL. Try again.')
        return

    account_id = get_steamid3(url)
    if account_id:
        user_id = message.chat.id
        await save_subscription(user_id, account_id)
        await message.answer(f'Subscription completed!')
    else:
        await message.answer('Failed to retrieve Steam ID. Check the URL.')
    await state.clear()

async def match_checker():
    while True:
        subscriptions = await get_all_subscriptions()
        for user_id, account_id, message_id in subscriptions:
            match_info = get_match_account_id(account_id)
            if match_info is not None:
                text = await generate_match_text(match_info)
                if message_id:
                    await bot.edit_message_text(text, user_id, message_id, parse_mode='HTML')
                else:
                    sent_message = await bot.send_message(user_id, text, parse_mode='HTML')
                    await update_message_id(user_id, account_id, sent_message.message_id)
        await asyncio.sleep(60)

async def generate_match_text(match_info):
    gbc = load('models/model.joblib')
    if not gbc:
        df = pd.read_csv('data/clean_data.csv')
        gbc, acc, _ = get_model(df)
    match_id, net_worth_team_0, net_worth_team_1, match_score, hero_ids_0, hero_ids_1 = get_match_info(match_info)
    heroes_0 = ', '.join(get_heroes(hero_ids_0))
    heroes_1 = ', '.join(get_heroes(hero_ids_1))
    
    softmax_arr = get_match_predict(match_info, gbc)
    ind = softmax_arr.argmax()
    if ind == 0:
        team = 'The Amber Hand'
    else:
        team = 'The Sapphire Flame'        
    prob = softmax_arr[0][ind]
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
'''
    return text


async def main():
    global db_pool
    db_pool = await create_db_pool()
    asyncio.create_task(match_checker())
    await dp.start_polling(bot)

async def close_db_pool():
    await db_pool.close()

atexit.register(lambda: asyncio.run(close_db_pool()))

if __name__ == '__main__':
    asyncio.run(main())